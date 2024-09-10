import asyncio
import atexit
import hashlib
import json
import os
import pickle
import random
import re
import sys
from abc import abstractmethod
from dataclasses import dataclass
# NOTE: if in py311 or lower, should install with `python -m pip install difflib`
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from loguru import logger
from metagpt.config2 import Config
from metagpt.llm import LLM
from metagpt.provider.base_llm import BaseLLM
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer, RobertaConfig, RobertaModel,
                          RobertaTokenizer)


def crop_code_lines(code: str | list[str], threshold: int) -> str | list[str]:  # simplified from source repo
    if isinstance(code, str):
        lines = code.split('\n')
        return "\n".join(lines[-threshold:]) if len(lines) > threshold else "\n".join(lines)

    if isinstance(code, list):
        cur_lines = -1
        for i in range(len(code) - 1, -1, -1):
            if "Ċ" in code[i]:
                cur_lines += 1
            if cur_lines == threshold:
                return code[i + 1:]
        return code


def accuracy_at_k(prediction_list: list[list[int]], golden_index_list: list[int], k: int) -> float:  # simplified from source repo
    if not golden_index_list:
        raise ValueError("The list of golden indices should not be empty.")
    if len(golden_index_list) != len(prediction_list):
        raise ValueError(f"Golden indices list and prediction list must have the same length, got {
                         len(golden_index_list)} and {len(prediction_list)}.")
    acc = sum(1 for i, index_list in enumerate(prediction_list) if golden_index_list[i] in index_list[:k])

    return round(acc / len(golden_index_list), 5)


# copy from repobench model/unixcoder
class UniXcoder(nn.Module):
    def __init__(self, model_name):
        """
            Build UniXcoder.

            Parameters:

            * `model_name`- huggingface model card name. e.g. microsoft/unixcoder-base
        """
        super(UniXcoder, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.config = RobertaConfig.from_pretrained(model_name)
        self.config.is_decoder = True
        self.model = RobertaModel.from_pretrained(model_name, config=self.config)

        self.register_buffer("bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024))
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.tokenizer.add_tokens(["<mask0>"], special_tokens=True)

    def tokenize(self, inputs, mode="<encoder-only>", max_length=512, padding=False):
        """
        Convert string to token ids

        Parameters:

        * `inputs`- list of input strings.
        * `max_length`- The maximum total source sequence length after tokenization.
        * `padding`- whether to pad source sequence length to max_length.
        * `mode`- which mode the sequence will use. i.e. <encoder-only>, <decoder-only>, <encoder-decoder>
        """
        assert mode in ["<encoder-only>", "<decoder-only>", "<encoder-decoder>"]
        assert max_length < 1024

        tokenizer = self.tokenizer

        tokens_ids = []
        for x in inputs:
            tokens = tokenizer.tokenize(x)
            if mode == "<encoder-only>":
                tokens = tokens[:max_length-4]
                tokens = [tokenizer.cls_token, mode, tokenizer.sep_token] + tokens + [tokenizer.sep_token]
            elif mode == "<decoder-only>":
                tokens = tokens[-(max_length-3):]
                tokens = [tokenizer.cls_token, mode, tokenizer.sep_token] + tokens
            else:
                tokens = tokens[:max_length-5]
                tokens = [tokenizer.cls_token, mode, tokenizer.sep_token] + tokens + [tokenizer.sep_token]

            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            if padding:
                tokens_id = tokens_id + [self.config.pad_token_id] * (max_length-len(tokens_id))
            tokens_ids.append(tokens_id)
        return tokens_ids

    def decode(self, source_ids):
        """ Convert token ids to string """
        predictions = []
        for x in source_ids:
            prediction = []
            for y in x:
                t = y.cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = self.tokenizer.decode(t, clean_up_tokenization_spaces=False)
                prediction.append(text)
            predictions.append(prediction)
        return predictions

    def forward(self, source_ids):
        """ Obtain token embeddings and sentence embeddings """
        mask = source_ids.ne(self.config.pad_token_id)
        token_embeddings = self.model(source_ids, attention_mask=mask.unsqueeze(1) * mask.unsqueeze(2))[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return token_embeddings, sentence_embeddings

    def generate(self, source_ids, decoder_only=True, eos_id=None, beam_size=5, max_length=64):
        """ Generate sequence given context (source_ids) """

        # Set encoder mask attention matrix: bidirectional for <encoder-decoder>, unirectional for <decoder-only>
        if decoder_only:
            mask = self.bias[:, :source_ids.size(-1), :source_ids.size(-1)]
        else:
            mask = source_ids.ne(self.config.pad_token_id)
            mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        if eos_id is None:
            eos_id = self.config.eos_token_id

        device = source_ids.device

        # Decoding using beam search
        preds = []
        zero = torch.LongTensor(1).fill_(0).to(device)
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        length = source_ids.size(-1)
        encoder_output = self.model(source_ids, attention_mask=mask)
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1, :, :source_len[i]].repeat(beam_size, 1, 1, 1) for x in y]
                       for y in encoder_output.past_key_values]
            beam = Beam(beam_size, eos_id, device)
            input_ids = beam.getCurrentState().clone()
            context_ids = source_ids[i:i+1, :source_len[i]].repeat(beam_size, 1)
            out = encoder_output.last_hidden_state[i:i+1, :source_len[i]].repeat(beam_size, 1, 1)
            for _ in range(max_length):
                if beam.done():
                    break
                if _ == 0:
                    hidden_states = out[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = beam.getCurrentState().clone()
                else:
                    length = context_ids.size(-1)+input_ids.size(-1)
                    out = self.model(input_ids, attention_mask=self.bias[:, context_ids.size(-1):length, :length],
                                     past_key_values=context).last_hidden_state
                    hidden_states = out[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState().clone()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero]*(max_length-len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)

        return preds


# copy from repobench model/unixcoder
class Beam(object):
    def __init__(self, size, eos, device):
        self.size = size
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.FloatTensor(size).zero_().to(device)
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [torch.LongTensor(size).fill_(0).to(device)]
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.nextYs[-1].view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = torch.div(bestScoresId, numWords, rounding_mode="floor")
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


@dataclass
class Settings:
    keep_lines: list[int]  # the lines to keep, e.g., [3, 10]


class CacheMixin:
    def __init__(self, cache_file: str = "/tmp/cache.pkl", save_frequency: int = 200) -> None:
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.call_count = 0
        self.save_frequency = save_frequency
        # Register the atexit function to save cache when the program exits
        atexit.register(self._save_cache_on_exit)

    def _generate_cache_key(self, code: str, candidates: list[str]) -> str:
        key = code + ''.join(candidates)
        return hashlib.md5(key.encode()).hexdigest()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                try:
                    return pickle.load(f)
                except (EOFError, pickle.UnpicklingError):
                    logger.warning("Failed to load cache file, starting with an empty cache.")
                    return {}
        else:
            return {}

    def _save_cache(self) -> None:
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def _save_cache_on_exit(self) -> None:
        logger.info(f"Saving cache to {self.cache_file} before program exits")
        self._save_cache()

    def _cache_retrieve(self, cache_key: str): return self.cache.get(cache_key)

    def _cache_store(self, cache_key: str, value: list[int]) -> None:
        self.cache[cache_key] = value
        self.call_count += 1
        if self.call_count >= self.save_frequency:
            logger.info(f"Saving cache to {self.cache_file} after {self.save_frequency} calls")
            self._save_cache()
            self.call_count = 0

    def __del__(self): self._save_cache_on_exit()


class Similarity:
    def retrieve(self, code: str, candidates: list[str]) -> list[int]:
        sim_scores = []
        for i, candidate in enumerate(candidates):
            sim_scores.append((i, self.similarity(code, candidate)))
        # sort the candidate index based on the edit similarity in a descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # only return the index
        ranks = [index for index, score in sim_scores]
        return ranks

    @abstractmethod
    def similarity(self, code1: str | list, code2: str | list) -> float: raise NotImplementedError


class AsyncSimilarity:
    async def retrieve(self, code: str, candidates: list[str]) -> list[int]:
        sim_scores = []
        for i, candidate in enumerate(candidates):
            sim_scores.append((i, self.similarity(code, candidate)))
        # sort the candidate index based on the edit similarity in a descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # only return the index
        ranks = [index for index, score in sim_scores]
        return ranks

    @abstractmethod
    async def similarity(self, code1: str | list, code2: str | list) -> float: raise NotImplementedError


class EditSimilarity(Similarity):
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")

    def similarity(self, code1: str | list, code2: str | list) -> float:
        # Check input types and tokenize as needed
        if isinstance(code1, str):
            assert self.tokenizer, "tokenizer must be provided if input is string"
            code1 = self.tokenizer.tokenize(code1)
        elif isinstance(code1, list):
            pass
        if isinstance(code2, str):
            assert self.tokenizer, "tokenizer must be provided if input is string"
            code2 = self.tokenizer.tokenize(code2)
        elif isinstance(code2, list):
            pass
        # compute and return the similarity ratio
        return SequenceMatcher(None, code1, code2).ratio()


class JaccardSimilarity(EditSimilarity):
    def similarity(self, code1: str | list, code2: str | list) -> float:
        # Check input types and tokenize/de-duplicate as needed
        if isinstance(code1, str):
            assert self.tokenizer, "tokenizer must be provided if input is string"
            code1 = set(self.tokenizer.tokenize(code1))
        elif isinstance(code1, list):
            code1 = set(code1)

        if isinstance(code2, str):
            assert self.tokenizer, "tokenizer must be provided if input is string"
            code2 = set(self.tokenizer.tokenize(code2))
        elif isinstance(code2, list):
            code2 = set(code2)
        try:
            return len(code1.intersection(code2)) / len(code1.union(code2))
        except ZeroDivisionError:
            logger.warning(f"ZeroDivisionError in {code1} & {code2}")
            return 0


class CosineSimilarityBase(Similarity):
    @abstractmethod
    def _get_max_length(self) -> int: raise NotImplementedError

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str | torch.device | int) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.max_length = self._get_max_length()
        self.model.to(torch.device(device))

    def get_embedding(self, code: str):
        if not code:
            return torch.zeros(self.model.config.hidden_size).to(self.model.device)
        code_tokens = self.tokenizer.tokenize(code, max_length=self.max_length, truncation=True)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        with torch.no_grad():
            # [1, seq_len, hidden_size]
            code_embeddings = self.model(torch.tensor(tokens_ids)[None, :].to(self.model.device))[0]
        code_embeddings = torch.mean(code_embeddings, dim=1)  # [1, hidden_size]
        code_embeddings = torch.squeeze(code_embeddings)  # [hidden_size]
        return code_embeddings

    def retrieve(self, code: str, candidates: list[str]) -> list[int]:
        code_embedding = self.get_embedding(code)
        candidates_embeddings = [self.get_embedding(candidate) for candidate in candidates]
        sim_scores = []
        for i, candidate_embedding in enumerate(candidates_embeddings):
            sim_scores.append((i, self.similarity(code_embedding, candidate_embedding)))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        ranks = [index for index, score in sim_scores]
        return ranks

    def similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        # check the input to be tensor
        assert isinstance(embedding1, torch.Tensor), "embedding1 must be a tensor"
        assert isinstance(embedding2, torch.Tensor), "embedding2 must be a tensor"
        # calculate the cosine similarity
        return torch.cosine_similarity(embedding1, embedding2, dim=0).item()


class CodegenSimilarity(CosineSimilarityBase):
    def _get_max_length(self) -> int: return 2048

    def __init__(self, model_name: str, device: str | torch.device | int = "cuda") -> None:
        assert "codegen" in model_name, "This class only for `codegen`"
        model = AutoModel.from_pretrained(model_name, cache_dir="cache")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        super().__init__(model, tokenizer, device)


class CodeGPTSimilarity(CosineSimilarityBase):
    def _get_max_length(self) -> int: return 512

    def __init__(self, model_name: str, device: str | torch.device | int = "cuda") -> None:
        assert "CodeGPT" in model_name, "This class only for `CodeGPT`"
        model = AutoModel.from_pretrained(model_name, cache_dir="cache")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id
        super().__init__(model, tokenizer, device)


class CodebertSimilarity(CosineSimilarityBase):
    def _get_max_length(self) -> int: return 512

    def __init__(self, model_name: str, device: str | torch.device | int = "cuda") -> None:
        assert "codebert" in model_name, "This class only for `codebert`"
        model = AutoModel.from_pretrained(model_name, cache_dir="cache")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        super().__init__(model, tokenizer, device)


class UnixcoderSimilarity(CosineSimilarityBase):
    def _get_max_length(self) -> int: return 512

    def __init__(self, model_name: str, device: str | torch.device | int = "cuda") -> None:
        assert "unixcoder" in model_name, "This class only for `unixcoder`"
        model = UniXcoder(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        super().__init__(model, tokenizer, device)
        self.model: UniXcoder

    def get_embedding(self, code: str):
        if not code:
            return torch.zeros(self.model.config.hidden_size).to(self.model.model.device)
        tokens_ids = self.model.tokenize([code], max_length=self.max_length, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(self.model.model.device)
        with torch.no_grad():
            _, code_embeddings = self.model(source_ids)  # [1, 768]
        code_embeddings = torch.squeeze(code_embeddings)  # [hidden_size]
        return code_embeddings


class VoyageSimilarity(Similarity, CacheMixin):
    def __init__(self, api_key: str, model: str = "voyage-code-2", cache_file: str = "/tmp/voyage_lru.pkl", save_frequency: int = 200) -> None:
        CacheMixin.__init__(self, cache_file, save_frequency)
        try:
            import voyageai
        except ImportError:
            logger.error("using `python -m pip install voyageai` at first")
            sys.exit(1)
        self.vo = voyageai.Client(api_key)
        self.model = model

    def retrieve(self, code: str, candidates: list[str]) -> list[int]:
        if code == "":
            sim_scores = [(i, len(candidate[:1000]) / 100) for i, candidate in enumerate(candidates)]
            sim_scores = sorted(sim_scores, key=lambda x: x[1])
        else:
            cache_key = self._generate_cache_key(code, candidates)
            cached_result = self._cache_retrieve(cache_key)
            if cached_result:
                logger.info("Cache hit")
                return cached_result
            code_embedding: list[float] = self.vo.embed([code], model=self.model, truncation=True).embeddings[0]
            # switch to get in queue to avoid total length over limit
            candidates_embeddings: list[list[float]] = [self.vo.embed([candidate],
                                                                      model=self.model,
                                                                      truncation=True).embeddings[0]
                                                        for candidate in candidates]
            sim_scores = [(i, self.similarity(code_embedding, candidate_embedding))
                          for i, candidate_embedding in enumerate(candidates_embeddings)]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        ranks = [index for index, _ in sim_scores]

        self._cache_store(cache_key, ranks)
        return ranks

    def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        return torch.cosine_similarity(torch.tensor(embedding1), torch.tensor(embedding2), dim=0).item()


class LLMSimilarityBase(AsyncSimilarity):
    @abstractmethod
    def _prompt(self): raise NotImplementedError

    def __init__(self, llm: BaseLLM) -> None:
        self.llm: BaseLLM = llm
        self.prompt: str = self._prompt()

    def clean_and_extract_numbers(self, result: str): return ' '.join(re.sub(r'[^0-9]', ' ', result).split())


class ListwiseLLMSimilarity(LLMSimilarityBase, CacheMixin):
    def _prompt(self) -> str: return """\
You will be given a code snippet and multiple candidates. Your task is to evaluate each candidate based on its relevance to the provided code. The higher the relevance between the candidate and the code, the higher the score should be. You must output an integer score between 0 and 100 for each candidate, where 0 is completely irrelevant and 100 is perfectly relevant. The scores must be separated by a single space, and no additional text should be included in your output.

Code:

{code}

Candidates:

{candidates}

Output the scores in the format: `score1 score2 score3 ...`
"""

    def __init__(self, llm: BaseLLM, cache_file: str = "/tmp/llm_listwise_cache.pkl", save_frequency: int = 200) -> None:
        LLMSimilarityBase.__init__(self, llm)
        CacheMixin.__init__(self, cache_file, save_frequency)

    async def retrieve(self, code: str, candidates: list[str]) -> list[int]:
        cache_key = self._generate_cache_key(code, candidates)
        if code == "":
            sim_scores = [(i, len(candidate[:1000]) / 100) for i, candidate in enumerate(candidates)]
            sim_scores = sorted(sim_scores, key=lambda x: x[1])
        else:
            cached_result = self._cache_retrieve(cache_key)
            if cached_result:
                logger.info("Cache hit")
                return cached_result
            sim_scores_bn = [0]*len(candidates)
            non_empty_indices = [index for index, string in enumerate(candidates) if string]
            non_empty_list = [candidates[i] for i in non_empty_indices]

            candidates_str = "\n\n".join([f"{i+1}.\n{cand}" for i, cand in enumerate(non_empty_list)])
            prompt = self.prompt.format(code=code, candidates=candidates_str)
            try:
                output = await self.llm.aask(msg=prompt)
            except:
                output = ""
            clear_output = self.clean_and_extract_numbers(output)
            ret = clear_output.split()
            if len(ret) == len(non_empty_list):
                sub_sim_scores = [int(i) for i in ret]
                for idx, sub_score in zip(non_empty_indices, sub_sim_scores):
                    sim_scores_bn[idx] = sub_score
            else:
                logger.warning(f"LLM({self.llm.__class__.__name__}) returned an unsupported response: \n{output}\n\
                                assuming all values are `0`")
            sim_scores = [(i, score) for i, score in enumerate(sim_scores_bn)]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        ranks = [index for index, score in sim_scores]

        self._cache_store(cache_key, ranks)
        return ranks


class PointwiseLLMSimilarity(LLMSimilarityBase, CacheMixin):
    def _prompt(self): return """\
You will be given a code snippet and one candidate. Your task is to evaluate the candidate based on its relevance to the provided code. The higher the relevance between the candidate and the code, the higher the score should be. You must output an integer score between 0 and 100, where 0 is completely irrelevant and 100 is perfectly relevant. Your output must only be a single number with no additional text.

Code:

{code}

Candidate:

{candidate}

Output the score as **a single integer**.
"""

    def __init__(self, llm: BaseLLM, cache_file: str = "/tmp/llm_pointwise_cache.pkl", save_frequency: int = 200) -> None:
        LLMSimilarityBase.__init__(self, llm)
        CacheMixin.__init__(self, cache_file, save_frequency)

    async def retrieve(self, code: str, candidates: list[str]) -> list[int]:
        if code == "":
            sim_scores = [(i, len(candidate[:1000]) / 100) for i, candidate in enumerate(candidates)]
            sim_scores = sorted(sim_scores, key=lambda x: x[1])
        else:
            sim_scores = []
            for ci, candidate in enumerate(candidates):
                cache_key = self._generate_cache_key(code, [candidate])
                if candidate == "":
                    score = 0
                else:
                    cached_result = self._cache_retrieve(cache_key)
                    if cached_result:
                        logger.info("Cache hit")
                        sim_scores.append((ci, cached_result[0]))
                        continue
                    prompt = self.prompt.format(code=code, candidate=candidate)
                    output = await self.llm.aask(msg=prompt)
                    clear_output = self.clean_and_extract_numbers(output)
                    ret = clear_output.split()
                    if len(ret) == 1:
                        score = int(ret[0])
                    else:
                        score = 0
                        logger.warning(f"LLM({self.llm.__class__.__name__}) returned an unsupported response: \n{output}\n\
                                        assuming `0`")
                sim_scores.append((ci, score))
                self._cache_store(cache_key, [score])
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        ranks = [index for index, score in sim_scores]
        return ranks


class Repobench:
    @abstractmethod
    def _get_languages(self): raise NotImplementedError

    def __init__(self, subsets: list[str], tasks: list[str], dataset_path: str) -> None:
        self.subsets: list[str] = subsets
        self.tasks: list[str] = tasks
        self.dataset_path: str = dataset_path

        self.datasets: dict[str, dict[str, Dataset]] = {}
        self.languages: list[str] = self._get_languages()

    def load_dataset(self):
        for subset in self.subsets:
            logger.info(f"loading dataset `{self.dataset_path} | {subset}`")
            dataset = load_dataset(self.dataset_path, subset, verification_mode="no_checks")
            self.datasets[subset] = {}
            for task in self.tasks:
                self.datasets[subset][task] = dataset[task]

    def load_data_parquet(self, samples_tag: int = 60):
        for subset in self.subsets:
            self.datasets[subset] = {}
            for task_name in self.tasks:
                self.datasets[subset][task_name] = load_dataset("parquet",
                                                                data_files=f"{self.dataset_path}/"
                                                                f"{subset}_{task_name}_sample_"
                                                                f"{samples_tag}.parquet")["train"]

    @abstractmethod
    def retrieve_test(self, settings: Settings, similarity: Similarity) -> dict[str, dict[str, list]]: raise NotImplementedError
    @abstractmethod
    def retrieve_eval(self, test_result: dict[str, dict[str, list]], print_random: bool) -> dict: raise NotImplementedError

    @abstractmethod
    async def async_retrieve_test(self, settings: Settings, similarity: AsyncSimilarity) -> dict[str, dict[str, list]]:
        raise NotImplementedError


class RepobenchRetriever(Repobench):
    def _get_languages(self): return ["python"]  # use `python` only

    def __init__(self) -> None:
        super().__init__(["python_cff", "python_cfr"],
                         ["test_easy", "test_hard"],
                         "tianyang/repobench-r")

    def retrieve_test(self, settings: Settings, similarity: Similarity) -> dict[str, dict[str, list]]:
        logger.info(f"retrieve in settings: {settings}, similarity using `{similarity.__class__.__name__}`")
        all_result: dict[str, dict[str, list]] = {}
        for subset_name, subset_data in self.datasets.items():
            logger.success(f"Processing `{subset_name}`")
            res: dict[str, list] = {}
            i = 0
            for task_name, task_set in subset_data.items():
                logger.success(f"In task `{task_name}`")
                res[task_name] = []
                for dic in tqdm(task_set, desc=f"running at {task_name}"):
                    res_dic = {}
                    for i in settings.keep_lines:
                        code = crop_code_lines(dic['code'], i)
                        candidates = dic['context']
                        res_dic[str(i)] = similarity.retrieve(code, candidates)
                    res_dic['ground_truth'] = dic['gold_snippet_index']
                    res[task_name].append(res_dic)
            all_result[subset_name] = res
        return all_result

    async def async_retrieve_test(self, settings: Settings, similarity: AsyncSimilarity) -> dict[str, dict[str, list]]:
        logger.info(f"async_retrieve in settings: {settings}, similarity using `{similarity.__class__.__name__}`")
        all_result: dict[str, dict[str, list]] = {}
        for subset_name, subset_data in self.datasets.items():
            logger.success(f"Processing `{subset_name}`")
            res: dict[str, list] = {}
            i = 0
            for task_name, task_set in subset_data.items():
                logger.success(f"In task `{task_name}`")
                res[task_name] = []
                for dic in tqdm(task_set, desc=f"running at {task_name}"):
                    res_dic = {}
                    for i in settings.keep_lines:
                        code = crop_code_lines(dic['code'], i)
                        candidates = dic['context']
                        res_dic[str(i)] = await similarity.retrieve(code, candidates)
                    res_dic['ground_truth'] = dic['gold_snippet_index']
                    res[task_name].append(res_dic)
            all_result[subset_name] = res
        return all_result

    def retrieve_eval(self, test_result: dict[str, dict[str, list]], print_random: bool = True) -> dict:
        record = self.__initialize_record()

        # loop through the datasets
        for subset_name, res in test_result.items():
            language, subset = subset_name.split("_")

            # Process test_easy and test_hard datasets
            for test_type in ['test_easy', 'test_hard']:
                self.__process_test(res[test_type], record, language, subset, test_type, print_random)

        self.__print_latex_results(record)
        return record

    def __initialize_record(self) -> dict:
        return {lang: {'cff': {'test_easy': {}, 'test_hard': {}},
                       'cfr': {'test_easy': {}, 'test_hard': {}}}
                for lang in self.languages}

    def __process_test(self, results, record, language, subset, test_type, print_random):
        gt, preds = [], {k: [] for k in ['3', '5', '10', '20', '30', '60', '120']}
        rd = [[] for _ in range(100)]

        for res_dic in results:
            gt.append(res_dic['ground_truth'])
            if print_random:
                for i in range(100):
                    rdm_get = list(range(len(res_dic['3'])))
                    random.shuffle(rdm_get)
                    rd[i].append(rdm_get)
            for k in preds.keys():
                if k in res_dic:
                    preds[k].append(res_dic[k])

        if print_random:
            self.__calculate_random_accuracy(rd, gt, record, language, subset, test_type)

        self.__calculate_accuracies(preds, gt, record, language, subset, test_type)

    def __calculate_random_accuracy(self, rd, gt, record, language, subset, test_type):
        rd_acc_at_1, rd_acc_at_3, rd_acc_at_5 = [], [], []
        for i in range(100):
            rd_acc_at_1.append(accuracy_at_k(rd[i], gt, k=1) * 100)
            rd_acc_at_3.append(accuracy_at_k(rd[i], gt, k=3) * 100)
            if test_type in ["test_easy", "test_hard"]:
                rd_acc_at_5.append(accuracy_at_k(rd[i], gt, k=5) * 100)
        if test_type in ["test_easy", "test_hard"]:
            rd_accs = [
                sum(rd_acc_at_1) / len(rd_acc_at_1),
                sum(rd_acc_at_3) / len(rd_acc_at_3),
                sum(rd_acc_at_5) / len(rd_acc_at_5)
            ]
        else:
            rd_accs = [
                sum(rd_acc_at_1) / len(rd_acc_at_1),
                sum(rd_acc_at_3) / len(rd_acc_at_3)
            ]
        record[language][subset][test_type]['rd'] = rd_accs

    def __calculate_accuracies(self, preds, gt, record, language, subset, test_type):
        for k, pred_list in preds.items():
            if pred_list:
                acc_at_1 = accuracy_at_k(pred_list, gt, k=1) * 100
                acc_at_3 = accuracy_at_k(pred_list, gt, k=3) * 100
                if test_type in ["test_easy", "test_hard"]:
                    acc_at_5 = accuracy_at_k(pred_list, gt, k=5) * 100
                    rec = [acc_at_1, acc_at_3, acc_at_5]
                else:
                    rec = [acc_at_1, acc_at_3]
                record[language][subset][test_type][k] = rec

    def __print_latex_results(self, record):
        for language in self.languages:
            logger.success(language.capitalize())
            if 'rd' in record[language]['cff']['test_easy']:
                prints = [*record[language]['cff']['test_easy']['rd'],
                          *record[language]['cfr']['test_easy']['rd'],
                          *record[language]['cff']['test_hard']['rd'],
                          *record[language]['cfr']['test_hard']['rd']]
                prints = [f"{i:.2f}" for i in prints]
                logger.success(f"rd & {' & '.join(prints)}")
            for i in ['3', '5', '10', '20', '30', '60', '120']:
                try:
                    prints = [*record[language]['cff']['test_easy'][i],
                              *record[language]['cfr']['test_easy'][i],
                              *record[language]['cff']['test_hard'][i],
                              *record[language]['cfr']['test_hard'][i]]
                    prints = [f"{i:.2f}" for i in prints]
                    logger.success(f"&{i} & {' & '.join(prints)}")
                except:
                    ...


class SubRepobenchRetriever(RepobenchRetriever):
    def __init__(self, dataset_path: str) -> None:
        Repobench.__init__(self,
                           ["python_cff", "python_cfr"],
                           ["test_easy", "test_hard"],
                           dataset_path)


def demo():
    settings = Settings(keep_lines=[3])  # NOTE: keep this setting in default
    # similarity = EditSimilarity()  # using "Salesforce/codegen-350M-multi" tokenizer
    # similarity = JaccardSimilarity()  # using "Salesforce/codegen-350M-multi" tokenizer
    # similarity = VoyageSimilarity(api_key=os.getenv("VOYAGE_KEY"))
    similarity = UnixcoderSimilarity(model_name="microsoft/unixcoder-base", device="cuda")
    benchmark = RepobenchRetriever()
    benchmark.load_dataset()
    test_result = benchmark.retrieve_test(settings, similarity)
    with open("results/retrieval/unixcoder-base_cosine/python2.json", "w") as f:
        json.dump(test_result, f)
    del test_result
    with open("results/retrieval/unixcoder-base_cosine/python2.json", "r") as f:
        test_result = json.load(f)
    eval_result = benchmark.retrieve_eval(test_result, print_random=True)
    logger.success(eval_result)


async def subset_tasks():
    save_out_dir = "test_results"
    settings = Settings(keep_lines=[3])  # NOTE: keep this setting in default
    benchmark = SubRepobenchRetriever(dataset_path="samples")
    benchmark.load_data_parquet(samples_tag=60)
    bench = benchmark.__class__.__name__
    os.makedirs(f"{save_out_dir}/{bench}", exist_ok=True)
    config2 = Path(os.path.join(os.path.expanduser("~"), ".metagpt", "config2.yaml"))
    llm_config = Config.from_yaml_file(config2).llm
    llm: BaseLLM = LLM(llm_config=llm_config)
    similarities: list[Similarity | AsyncSimilarity] = [
        EditSimilarity(),  # in paper
        JaccardSimilarity(),  # in paper
        UnixcoderSimilarity(model_name="microsoft/unixcoder-base", device="cuda"),  # in paper
        ListwiseLLMSimilarity(llm=llm),
        PointwiseLLMSimilarity(llm=llm),
        VoyageSimilarity(api_key=os.getenv("VOYAGE_KEY")),
    ]

    for similarity in similarities:
        simi = similarity.__class__.__name__
        logger.success(f"TESTING Similarity = {simi}")
        if os.path.exists(f"{save_out_dir}/{bench}/{simi}.retrieve_test.json"):
            with open(f"{save_out_dir}/{bench}/{simi}.retrieve_test.json", "r") as f:
                test_result = json.load(f)
            logger.success(f"Loaded Similarity = {simi} test_result = `{save_out_dir}/{bench}/{simi}.retrieve_test.json`")
        else:
            if issubclass(similarity.__class__, AsyncSimilarity):
                test_result = await benchmark.async_retrieve_test(settings, similarity)
            elif issubclass(similarity.__class__, Similarity):
                test_result = benchmark.retrieve_test(settings, similarity)
            else:
                logger.info(f"not support Similarity type `{simi}`")
                continue
            with open(f"{save_out_dir}/{bench}/{simi}.retrieve_test.json", "w") as f:
                json.dump(test_result, f, indent=4)
            logger.success(f"Writed Similarity = {simi} test_result = `{save_out_dir}/{bench}/{simi}.retrieve_test.json`")

        eval_result = benchmark.retrieve_eval(test_result, print_random=True)
        logger.success(eval_result)
        with open(f"{save_out_dir}/{bench}/{simi}.retrieve_eval.json", "w") as f:
            json.dump(eval_result,  f, indent=4)
        logger.success(f"Writed Similarity = {simi} eval_result = `{save_out_dir}/{bench}/{simi}.retrieve_eval.json`")
        logger.success(f"FINISH {simi}")


if __name__ == "__main__":
    asyncio.run(subset_tasks())
