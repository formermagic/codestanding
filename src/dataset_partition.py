"""Partition given dataset into train/val/test subsets.

Usage:
    dataset_partition.py split \
        --dataset-prefix=<path> \
        --extensions=<ext> \
        --split-ratio=<ratio> \
        --output-path=<output>

Options:
    --dataset-prefix=<path>     A common prefix for the dataset path.
    --extensions=<ext>          A comma separated string with file extensions,
                                requires 2 separated arguments (source, target).
    --split-ratio=<ratio>       A comma separated string with partition split ratio,
                                requires 3 separated arguments (train, val, test).
                                (Example: "0.8, 0.15, 0.05").
    --output-path=<output>      An output path to write results to.
"""
import linecache
import math
import os
import typing

import numpy as np
from docopt import docopt
from tqdm import tqdm

from .utils import lines_in_file, parse_listed_arg


class TranslationDatasetPartition:
    def __init__(
        self, dataset_prefix: str, extensions: typing.Tuple[str, str]
    ) -> None:
        self.dataset_prefix = dataset_prefix
        self.extensions = extensions
        self.source_filepath = dataset_prefix + f".{extensions[0]}"
        self.target_filepath = dataset_prefix + f".{extensions[1]}"

    def split(
        self, ratio: typing.Tuple[float, float, float], output_path: str
    ) -> None:
        source_n_lines = lines_in_file(self.source_filepath)
        target_n_lines = lines_in_file(self.target_filepath)
        assert (
            source_n_lines == target_n_lines
        ), "Number of lines must match in both files!"

        n_lines = source_n_lines
        indices = np.arange(n_lines)
        train, val, test = self.partition_chunk(indices, ratio)

        _ = linecache.getline(self.source_filepath, 1)
        _ = linecache.getline(self.target_filepath, 1)

        train_prefix = os.path.join(output_path, "train")
        val_prefix = os.path.join(output_path, "val")
        test_prefix = os.path.join(output_path, "test")

        self.save_chunk(train, train_prefix)
        self.save_chunk(val, val_prefix)
        self.save_chunk(test, test_prefix)

    def partition_chunk(
        self, chunk: np.ndarray, ratio: typing.Tuple[float, float, float]
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert math.ceil(sum(ratio)) == 1, "Split ratio must sum up to 1"

        train_len = int(chunk.shape[0] * ratio[0])
        val_len = int(chunk.shape[0] * ratio[1])
        shuffled_chunk = np.random.permutation(chunk)

        train = shuffled_chunk[:train_len]
        val = shuffled_chunk[train_len : train_len + val_len]
        test = shuffled_chunk[train_len + val_len :]

        return train, val, test

    def save_chunk(self, chunk: np.ndarray, filepath_prefix: str) -> None:
        output_path = os.path.dirname(filepath_prefix)
        os.makedirs(output_path, exist_ok=True)

        source_filepath = filepath_prefix + f".{self.extensions[0]}"
        target_filepath = filepath_prefix + f".{self.extensions[1]}"
        source_file = open(source_filepath, mode="w")
        target_file = open(target_filepath, mode="w")

        kwargs = {
            "total": chunk.shape[0],
            "unit": "it",
            "unit_scale": True,
            "leave": True,
        }

        with source_file, target_file:
            for idx in tqdm(chunk, **kwargs):
                source_file.write(linecache.getline(self.source_filepath, idx))
                target_file.write(linecache.getline(self.target_filepath, idx))


def parse_split_ratio(arg: str) -> typing.Tuple[float, float, float]:
    listed_args = parse_listed_arg(arg)
    split_ratio = [float(val) for val in listed_args[:3]]
    return split_ratio


def parse_extensions(arg: str) -> typing.Tuple[str, str]:
    listed_args = parse_listed_arg(arg)
    return listed_args[:2]


def main() -> None:
    """
    python -m src.dataset_partition split \
        --dataset-prefix=/workspace/tmp/code2ast/test_dataset/all \
        --extensions="src, ast" \
        --split-ratio="0.8, 0.15, 0.05" \
        --output-path=/workspace/tmp/code2ast/test_dataset.split
    """
    arguments = docopt(__doc__, version="Dataset Partition 1.0")

    if arguments["split"]:
        dataset_prefix = str(arguments["--dataset-prefix"])
        extensions = parse_extensions(arguments["--extensions"])
        split_ratio = parse_split_ratio(arguments["--split-ratio"])
        output_path = str(arguments["--output-path"])

        dataset_partition = TranslationDatasetPartition(
            dataset_prefix, extensions,
        )

        dataset_partition.split(split_ratio, output_path)


if __name__ == "__main__":
    main()
