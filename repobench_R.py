from dataclasses import dataclass
# NOTE: if in py311 or lower, should install with `python -m pip install difflib`
from difflib import SequenceMatcher
import random

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from loguru import logger
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


def accuracy_at_k(prediction_list, golden_index_list, k):
    """
    This function computes the accuracy at k. It returns a float value between 0 and 1 indicating the
    accuracy at k, where a value of 1 means the correct code is retrieved at the top k positions and
    a value of 0 means the correct code is not retrieved at the top k positions.

    Args:
    prediction_list: list, a list of lists, where each list contains the indices of the retrieved codes.
    golden_index_list: list, a list of integers, where each integer is the index of the correct code.
    k: int, the number of retrieved codes.

    Returns:
    Float, the accuracy at k.
    """

    if len(golden_index_list) == 0:
        raise ValueError("The list of golden indices should not be empty.")

    assert len(golden_index_list) == len(prediction_list), \
        "The length of the golden indices list should be equal to the length of the prediction list, however, " \
        f"the length of the golden indices list is {len(golden_index_list)} and the length of the prediction list is {
        len(prediction_list)}."

    acc = 0

    for i in range(len(prediction_list)):
        golden_index = golden_index_list[i]
        index_list = prediction_list[i]

        if len(index_list) < k:
            raise ValueError("The number of retrieved codes should be greater than k.")

        top_k_indices = index_list[:k]

        if golden_index not in top_k_indices:
            continue
        else:
            acc += 1

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

    def similarity(self, code1: str | list, code2: str | list) -> float: raise NotImplementedError


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


class Repobench:
    def __init__(self, subsets: list[str], tasks: list[str], dataset_path: str) -> None:
        self.subsets: list[str] = subsets
        self.tasks: list[str] = tasks
        self.dataset_path: str = dataset_path

        self.datasets: dict[str, dict[str, Dataset]] = {}

    def load_dataset(self):
        for subset in self.subsets:
            dataset = load_dataset(self.dataset_path, subset, ignore_verifications=True, verification_mode="no_checks")
            self.datasets[subset] = {}
            for task in self.tasks:
                self.datasets[subset][task] = dataset[task]

    def retrieve_test(self) -> dict[str, dict[str, list]]: raise NotImplementedError

    def retrieve_eval(self, test_result: dict[str, dict[str, list]], print_random: bool) -> dict: raise NotImplementedError


class RepobenchRetriever(Repobench):
    def __init__(self) -> None:
        super().__init__(["python_cff", "python_cfr"],
                         ["test_easy", "test_hard"],
                         "tianyang/repobench-r")

    def retrieve_test(self, settings: Settings, similarity: Similarity) -> dict[str, dict[str, list]]:
        all_result: dict[str, dict[str, list]] = {}
        for subset_name, subset_data in self.datasets.items():
            logger.success(f"Processing {subset_name}")
            res: dict[str, list] = {}
            i = 0
            for task_name, task_set in subset_data.items():
                logger.success(f"In task {task_name}")
                res[task_name] = []
                for dic in tqdm(task_set, desc=f"running {task_name}"):
                    res_dic = {}
                    for i in settings.keep_lines:
                        code = crop_code_lines(dic['code'], i)
                        candidates = dic['context']
                        res_dic[i] = similarity.retrieve(code, candidates)
                    res_dic['ground_truth'] = dic['gold_snippet_index']
                    res[task_name].append(res_dic)
            all_result[subset_name] = res
        return all_result

    def retrieve_eval(self, test_result: dict[str, dict[str, list]], print_random: bool = True) -> dict:
        record = {
            'python': {
                'cff': {
                    'test_easy': {},
                    'test_hard': {}
                },
                'cfr': {
                    'test_easy': {},
                    'test_hard': {}
                }
            },
            'java': {
                'cff': {
                    'test_easy': {},
                    'test_hard': {}
                },
                'cfr': {
                    'test_easy': {},
                    'test_hard': {}
                }
            }
        }

        # loop through the files
        for subset_name, res in test_result.items():

            # get the language, and subset
            language = subset_name.split("_")[0]
            subset = subset_name.split("_")[-1].split(".")[0]

            # test the easy retrieval
            gt, pred_3, pred_5, pred_10, pred_20, pred_30, pred_60, pred_120 = [], [], [], [], [], [], [], []
            # create a list with 10 empty lists
            rd = [[] for _ in range(100)]
            for res_dic in res['test_easy']:
                gt.append(res_dic['ground_truth'])
                if print_random:
                    for i in range(100):
                        rdm_get = list(range(len(res_dic['3'])))
                        random.shuffle(rdm_get)
                        rd[i].append(rdm_get)
                if '3' in res_dic:
                    pred_3.append(res_dic['3'])
                if '5' in res_dic:
                    pred_5.append(res_dic['5'])
                if '10' in res_dic:
                    pred_10.append(res_dic['10'])
                if '20' in res_dic:
                    pred_20.append(res_dic['20'])
                if '30' in res_dic:
                    pred_30.append(res_dic['30'])
                if '60' in res_dic:
                    pred_60.append(res_dic['60'])
                if '120' in res_dic:
                    pred_120.append(res_dic['120'])

            # # average of the 10 random lists
            if print_random:
                easy_rd_acc_at_1, easy_rd_acc_at_3 = [], []
                for i in range(100):
                    easy_rd_acc_at_1.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=1)*100)
                    easy_rd_acc_at_3.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=3)*100)

                # # average of the 10 random lists
                easy_rd_acc_at_1 = sum(easy_rd_acc_at_1)/len(easy_rd_acc_at_1)
                easy_rd_acc_at_3 = sum(easy_rd_acc_at_3)/len(easy_rd_acc_at_3)
                record[language][subset]['test_easy']['rd'] = [easy_rd_acc_at_1, easy_rd_acc_at_3]

            if '3' in res_dic:
                easy_3_acc_at_1 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=1)*100
                easy_3_acc_at_3 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=3)*100
                record[language][subset]['test_easy']['3'] = [easy_3_acc_at_1, easy_3_acc_at_3]
            if '5' in res_dic:
                easy_5_acc_at_1 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=1)*100
                easy_5_acc_at_3 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=3)*100
                record[language][subset]['test_easy']['5'] = [easy_5_acc_at_1, easy_5_acc_at_3]
            if '10' in res_dic:
                easy_10_acc_at_1 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=1)*100
                easy_10_acc_at_3 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=3)*100
                record[language][subset]['test_easy']['10'] = [easy_10_acc_at_1, easy_10_acc_at_3]
            if '20' in res_dic:
                easy_20_acc_at_1 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=1)*100
                easy_20_acc_at_3 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=3)*100
                record[language][subset]['test_easy']['20'] = [easy_20_acc_at_1, easy_20_acc_at_3]
            if '30' in res_dic:
                easy_30_acc_at_1 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=1)*100
                easy_30_acc_at_3 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=3)*100
                record[language][subset]['test_easy']['30'] = [easy_30_acc_at_1, easy_30_acc_at_3]
            if '60' in res_dic:
                easy_60_acc_at_1 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=1)*100
                easy_60_acc_at_3 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=3)*100
                record[language][subset]['test_easy']['60'] = [easy_60_acc_at_1, easy_60_acc_at_3]
            if '120' in res_dic:
                easy_120_acc_at_1 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=1)*100
                easy_120_acc_at_3 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=3)*100
                record[language][subset]['test_easy']['120'] = [easy_120_acc_at_1, easy_120_acc_at_3]

            # test the hard dataset
            gt, pred_3, pred_5, pred_10, pred_20, pred_30, pred_60, pred_120 = [], [], [], [], [], [], [], []
            rd = [[] for _ in range(100)]
            for res_dic in res['test_hard']:
                gt.append(res_dic['ground_truth'])
                if print_random:
                    for i in range(100):
                        rdm_get = list(range(len(res_dic['3'])))
                        random.shuffle(rdm_get)
                        rd[i].append(rdm_get)
                if '3' in res_dic:
                    pred_3.append(res_dic['3'])
                if '5' in res_dic:
                    pred_5.append(res_dic['5'])
                if '10' in res_dic:
                    pred_10.append(res_dic['10'])
                if '20' in res_dic:
                    pred_20.append(res_dic['20'])
                if '30' in res_dic:
                    pred_30.append(res_dic['30'])
                if '60' in res_dic:
                    pred_60.append(res_dic['60'])
                if '120' in res_dic:
                    pred_120.append(res_dic['120'])

            # # average of the 10 random lists

            if print_random:
                hard_rd_acc_at_1, hard_rd_acc_at_3, hard_rd_acc_at_5 = [], [], []
                for i in range(100):
                    hard_rd_acc_at_1.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=1)*100)
                    hard_rd_acc_at_3.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=3)*100)
                    hard_rd_acc_at_5.append(accuracy_at_k(prediction_list=rd[i], golden_index_list=gt, k=5)*100)

                hard_rd_acc_at_1 = sum(hard_rd_acc_at_1)/len(hard_rd_acc_at_1)
                hard_rd_acc_at_3 = sum(hard_rd_acc_at_3)/len(hard_rd_acc_at_3)
                hard_rd_acc_at_5 = sum(hard_rd_acc_at_5)/len(hard_rd_acc_at_5)
                record[language][subset]['test_hard']['rd'] = [hard_rd_acc_at_1, hard_rd_acc_at_3, hard_rd_acc_at_5]

            if '3' in res_dic:
                hard_3_acc_at_1 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=1)*100
                hard_3_acc_at_3 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=3)*100
                hard_3_acc_at_5 = accuracy_at_k(prediction_list=pred_3, golden_index_list=gt, k=5)*100
                record[language][subset]['test_hard']['3'] = [hard_3_acc_at_1, hard_3_acc_at_3, hard_3_acc_at_5]
            if '5' in res_dic:
                hard_5_acc_at_1 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=1)*100
                hard_5_acc_at_3 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=3)*100
                hard_5_acc_at_5 = accuracy_at_k(prediction_list=pred_5, golden_index_list=gt, k=5)*100
                record[language][subset]['test_hard']['5'] = [hard_5_acc_at_1, hard_5_acc_at_3, hard_5_acc_at_5]
            if '10' in res_dic:
                hard_10_acc_at_1 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=1)*100
                hard_10_acc_at_3 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=3)*100
                hard_10_acc_at_5 = accuracy_at_k(prediction_list=pred_10, golden_index_list=gt, k=5)*100
                record[language][subset]['test_hard']['10'] = [hard_10_acc_at_1, hard_10_acc_at_3, hard_10_acc_at_5]
            if '20' in res_dic:
                hard_20_acc_at_1 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=1)*100
                hard_20_acc_at_3 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=3)*100
                hard_20_acc_at_5 = accuracy_at_k(prediction_list=pred_20, golden_index_list=gt, k=5)*100
                record[language][subset]['test_hard']['20'] = [hard_20_acc_at_1, hard_20_acc_at_3, hard_20_acc_at_5]
            if '30' in res_dic:
                hard_30_acc_at_1 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=1)*100
                hard_30_acc_at_3 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=3)*100
                hard_30_acc_at_5 = accuracy_at_k(prediction_list=pred_30, golden_index_list=gt, k=5)*100
                record[language][subset]['test_hard']['30'] = [hard_30_acc_at_1, hard_30_acc_at_3, hard_30_acc_at_5]
            if '60' in res_dic:
                hard_60_acc_at_1 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=1)*100
                hard_60_acc_at_3 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=3)*100
                hard_60_acc_at_5 = accuracy_at_k(prediction_list=pred_60, golden_index_list=gt, k=5)*100
                record[language][subset]['test_hard']['60'] = [hard_60_acc_at_1, hard_60_acc_at_3, hard_60_acc_at_5]
            if '120' in res_dic:
                hard_120_acc_at_1 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=1)*100
                hard_120_acc_at_3 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=3)*100
                hard_120_acc_at_5 = accuracy_at_k(prediction_list=pred_120, golden_index_list=gt, k=5)*100
                record[language][subset]['test_hard']['120'] = [hard_120_acc_at_1, hard_120_acc_at_3, hard_120_acc_at_5]

        # print latex
        print("Python")
        # print rd as baseline
        if 'rd' in record['python']['cff']['test_easy']:
            print(f" rd & {record['python']['cff']['test_easy']['rd'][0]:.2f} & {record['python']['cff']['test_easy']['rd'][1]:.2f} & {record['python']['cfr']['test_easy']['rd'][0]:.2f} & {record['python']['cfr']['test_easy']['rd'][1]:.2f} & {record['python']['cff']['test_hard']['rd'][0]:.2f} & {
                record['python']['cff']['test_hard']['rd'][1]:.2f} & {record['python']['cff']['test_hard']['rd'][2]:.2f}& {record['python']['cfr']['test_hard']['rd'][0]:.2f} & {record['python']['cfr']['test_hard']['rd'][1]:.2f} & {record['python']['cfr']['test_hard']['rd'][2]:.2f} \\\\")
        for i in [3, 5, 10, 20, 30, 60, 120]:
            try:
                print(f"&{i} & {record['python']['cff']['test_easy'][str(i)][0]:.2f} & {record['python']['cff']['test_easy'][str(i)][1]:.2f} & {record['python']['cfr']['test_easy'][str(i)][0]:.2f} & {record['python']['cfr']['test_easy'][str(i)][1]:.2f} & {record['python']['cff']['test_hard'][str(i)][0]:.2f} & {
                    record['python']['cff']['test_hard'][str(i)][1]:.2f} & {record['python']['cff']['test_hard'][str(i)][2]:.2f}& {record['python']['cfr']['test_hard'][str(i)][0]:.2f} & {record['python']['cfr']['test_hard'][str(i)][1]:.2f} & {record['python']['cfr']['test_hard'][str(i)][2]:.2f} \\\\")
            except:
                pass

        print("Java")
        # print rd as baseline
        if 'rd' in record['java']['cff']['test_easy']:
            print(f" rd & {record['java']['cff']['test_easy']['rd'][0]:.2f} & {record['java']['cff']['test_easy']['rd'][1]:.2f} & {record['java']['cfr']['test_easy']['rd'][0]:.2f} & {record['java']['cfr']['test_easy']['rd'][1]:.2f} & {record['java']['cff']['test_hard']['rd'][0]:.2f} & {
                record['java']['cff']['test_hard']['rd'][1]:.2f} & {record['java']['cff']['test_hard']['rd'][2]:.2f}& {record['java']['cfr']['test_hard']['rd'][0]:.2f} & {record['java']['cfr']['test_hard']['rd'][1]:.2f} & {record['java']['cfr']['test_hard']['rd'][2]:.2f} \\\\")
        for i in [3, 5, 10, 20, 30, 60, 120]:
            try:
                print(f"&{i} & {record['java']['cff']['test_easy'][str(i)][0]:.2f} & {record['java']['cff']['test_easy'][str(i)][1]:.2f} & {record['java']['cfr']['test_easy'][str(i)][0]:.2f} & {record['java']['cfr']['test_easy'][str(i)][1]:.2f} & {record['java']['cff']['test_hard'][str(i)][0]:.2f} & {
                    record['java']['cff']['test_hard'][str(i)][1]:.2f} & {record['java']['cff']['test_hard'][str(i)][2]:.2f}& {record['java']['cfr']['test_hard'][str(i)][0]:.2f} & {record['java']['cfr']['test_hard'][str(i)][1]:.2f} & {record['java']['cfr']['test_hard'][str(i)][2]:.2f} \\\\")
            except:
                pass

        return record


def demo():
    settings = Settings(keep_lines=[3])
    # similarity = EditSimilarity()  # using "Salesforce/codegen-350M-multi" tokenizer
    # similarity = JaccardSimilarity()  # using "Salesforce/codegen-350M-multi" tokenizer
    similarity = UnixcoderSimilarity(model_name="microsoft/unixcoder-base", device="cuda")
    benchmark = RepobenchRetriever()
    test_result = benchmark.retrieve_test(settings, similarity)
    eval_result = benchmark.retrieve_eval(test_result, print_random=True)


if __name__ == "__main__":
    demo()
