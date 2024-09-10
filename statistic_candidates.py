import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from run_repobench_r import Dataset, load_data, load_data_parquet
from tqdm import tqdm


def statistic(dataset: Dataset):
    lines: list[int] = []
    counts: list[int] = []
    str_length: list[int] = []
    for dic in tqdm(dataset, desc=f"loading..."):
        candidates: list[str] = dic['context']
        counts.append(len(candidates))
        for candidate in candidates:
            str_length.append(len(candidate))
            lines.append(len(candidate.split("\n")))

    return lines, counts, str_length


def draw(data: list[int], filename: str, xlabel: str, log=False):
    plt.hist(data, bins=range(min(data), max(data) + 2), edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Data Distribution (average value = {np.mean(data)})')
    if log:
        plt.yscale('log')
    plt.savefig(f"statics/{filename}.png", format='png')
    plt.close()


def main():
    # datasets = load_data()
    datasets = load_data_parquet()
    for subset_name, subset_data in datasets.items():
        logger.success(f"Processing {subset_name}")
        for task_name, task_set in subset_data.items():
            logger.success(f"In task {task_name}")
            lines, counts, str_length = statistic(task_set)
            draw(lines, f"samples_{subset_name}_{task_name}_lines",
                 xlabel=f"{subset_name}_{task_name}_lines", log=True)

            draw(str_length, f"samples_{subset_name}_{task_name}_str_length",
                 xlabel=f"{subset_name}_{task_name}_str_length", log=True)

            draw(counts, f"samples_{subset_name}_{task_name}_candidate_counts",
                 xlabel=f"{subset_name}_{task_name}_candidate_counts")


if __name__ == "__main__":
    main()
