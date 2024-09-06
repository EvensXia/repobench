

import json
import os

from archive_data.utils import crop_code_lines
from datasets import Dataset, load_dataset
from loguru import logger
from model.unixcoder import UniXcoder
from retriever.retriever import retrieve
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_data(dataset_path: str = "tianyang/repobench-r",
              tasks: list[str] = ["test_easy", "test_hard"],
              subsets: list[str] = ["python_cff", "python_cfr"]) -> dict[str, dict[str, Dataset]]:
    datasets = {}
    for subset in subsets:
        dataset = load_dataset(dataset_path, subset, ignore_verifications=True, verification_mode="no_checks")
        datasets[subset] = {}
        for task in tasks:
            datasets[subset][task] = dataset[task]

    return datasets


def main(
    similarity: str,  # the similarity used to retrieve, e.g., cosine, edit, jaccard
    keep_lines: list,  # the lines to keep, e.g., [3, 10]
    model_name: str = "",  # the model used to encode the code, e.g., microsoft/unixcoder-base
    max_length: int = 512,  # max length of the code
):
    # load the data
    datasets = load_data()

    # defualt lexical retrieval, no need to load the model
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi", cache_dir="cache")
    model = None

    # if semantic retrieval
    if model_name:
        # load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="cache")
        if "codegen" in model_name:
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
            max_length = 2048
        elif "CodeGPT" in model_name:
            max_length = 512
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif "codebert" in model_name:
            max_length = 512
        elif "unixcoder" in model_name:
            max_length = 512

        if "unixcoder" in model_name:
            model = UniXcoder(model_name)
        else:
            model = AutoModel.from_pretrained(model_name, cache_dir="cache")
        model.to("cuda")

    for subset_name, subset_data in datasets.items():
        logger.success(f"Processing {subset_name}")
        res: dict[str, list] = {}
        i = 0
        for task_name, task_set in subset_data.items():
            logger.success(f"In task {task_name}")
            res[task_name] = []
            for dic in tqdm(task_set, desc=f"running {task_name}"):
                res_dic = {}
                for i in keep_lines:
                    code = crop_code_lines(dic['code'], i)
                    candidates = dic['context']
                    res_dic[i] = retrieve(
                        code=code,
                        candidates=candidates,
                        tokenizer=tokenizer,
                        model=model,
                        max_length=max_length,
                        similarity=similarity)
                res_dic['ground_truth'] = dic['gold_snippet_index']
                res[task_name].append(res_dic)
        # write
        if model_name:
            os.makedirs(f'results/retrieval/{model_name.split("/")[-1]}', exist_ok=True)
            with open(f"results/retrieval/{model_name.split('/')[-1]}/{subset_name}.json", "w") as f:
                json.dump(res, f, indent=4)
        else:
            os.makedirs(f'results/retrieval/{similarity}', exist_ok=True)
            with open(f"results/retrieval/{similarity}/{subset_name}.json", "w") as f:
                json.dump(res, f, indent=4)


if __name__ == "__main__":
    # import fire
    # fire.Fire(main)
    main(
        similarity="edit",  # "edit", "jaccard", "cosine"
        keep_lines=[3],
        model_name="microsoft/unixcoder-base",
        max_length=512
    )
