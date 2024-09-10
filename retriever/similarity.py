from fuzzywuzzy import fuzz
from transformers import AutoTokenizer
from difflib import SequenceMatcher
import torch
from typing import Union
import math


def edit_similarity(
    code1: Union[str, list],
    code2: Union[str, list],
    tokenizer: AutoTokenizer = None
) -> float:

    # Check input types and tokenize as needed
    if isinstance(code1, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code1 = tokenizer.tokenize(code1)
    elif isinstance(code1, list):
        pass

    if isinstance(code2, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code2 = tokenizer.tokenize(code2)
    elif isinstance(code2, list):
        pass

    # compute and return the similarity ratio
    return SequenceMatcher(None, code1, code2).ratio()


def jaccard_similarity(
    code1: Union[str, list],
    code2: Union[str, list],
    tokenizer: AutoTokenizer = None
) -> float:

    # Check input types and tokenize/de-duplicate as needed
    if isinstance(code1, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code1 = set(tokenizer.tokenize(code1))
    elif isinstance(code1, list):
        code1 = set(code1)

    if isinstance(code2, str):
        assert tokenizer, "tokenizer must be provided if input is string"
        code2 = set(tokenizer.tokenize(code2))
    elif isinstance(code2, list):
        code2 = set(code2)

    try:
        return len(code1.intersection(code2)) / len(code1.union(code2))
    except ZeroDivisionError:
        print("ZeroDivisionError")
        print(code1, code2)
        return 0


def cosine_similarity(
        embedding1,
        embedding2
):
    """
    Calculate the cosine similarity between two embeddings
    """
    # check the input to be tensor
    assert isinstance(embedding1, torch.Tensor), "embedding1 must be a tensor"
    assert isinstance(embedding2, torch.Tensor), "embedding2 must be a tensor"

    # calculate the cosine similarity
    return torch.cosine_similarity(embedding1, embedding2, dim=0).item()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
    test1 = ["", ["sdasd", "", "dassfasdg4"]]
    test2 = ["sdasd", ["sdasd", "", "dassfasdg4"]]
    test3 = ["sdasd", ["sdasd", "", "dassfasdg4"]]
    test4 = ["sdasd", ["sdasd", "dassfasdg4"]]
    for i, test in enumerate([test1, test2, test3, test4]):
        code = test[0]
        candidate = test[1]
        print(f"test_{i+1}", code, candidate)
        print("jaccard_similarity:", [jaccard_similarity(code, candidatei, tokenizer) for candidatei in candidate])
        print("edit_similarity:", [edit_similarity(code, candidatei, tokenizer) for candidatei in candidate])
        print("")
