import json
import random

from loguru import logger
from run_repobench_r import Dataset, load_data


def sample_dataset(dataset: Dataset, selects: int = 60):
    selected_index = random.sample(range(len(dataset)), k=selects)
    selected_dataset = dataset.select(selected_index)
    return selected_index, selected_dataset


def main():
    SAMPLES = 60
    datasets = load_data()
    indexes = {}
    for subset_name, subset_data in datasets.items():
        logger.success(f"Processing {subset_name}")
        indexes[subset_name] = {}
        for task_name, task_set in subset_data.items():
            logger.success(f"In task {task_name}")
            selected_index, selected_dataset = sample_dataset(task_set, selects=SAMPLES)
            selected_dataset.to_parquet(f"samples/{subset_name}_{task_name}_sample_{SAMPLES}.parquet")
            indexes[subset_name][task_name] = {}
            indexes[subset_name][task_name]["index"] = selected_index
            indexes[subset_name][task_name]["repo_name"] = selected_dataset['repo_name']
    with open("samples_indexes.json", "w") as f:
        json.dump(indexes, f, indent=4)


if __name__ == "__main__":
    main()
