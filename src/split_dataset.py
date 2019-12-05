"""Split a preprocessed and tokenized dataset
(a tuple of source+target files) into train/val/test
buckets for further processing tasks.

Usage:
    split_dataset.py split \
            --dataset_prefix=<x> \
            --exts=<ext,ext> \
            --split-ratio=<s,s,s> \
            --dest-path=<dest>
"""
import os
import typing

from docopt import docopt
from torchtext.data import Dataset, Field
from torchtext.datasets import TranslationDataset


def translation_dataset(
    dataset_prefix: str,
    exts: typing.Tuple[str, str],
    split_ratio: typing.Tuple[float, float, float],
) -> typing.Tuple[Dataset, Dataset, Dataset]:
    assert (
        sum(split_ratio) == 1.0
    ), "Split ratio is a set of 3 numbers that sum up to 1.0"

    src_field = Field(init_token="<s>", eos_token="</s>", include_lengths=True)
    trg_field = Field(init_token="<s>", eos_token="</s>")
    data_fields = [("source", src_field), ("target", trg_field)]

    dataset = TranslationDataset(
        path=dataset_prefix, exts=exts, fields=data_fields, filter_pred=None,
    )

    train_data, test_data, valid_data = dataset.split(split_ratio=split_ratio)

    return train_data, valid_data, test_data


def save_dataset(dataset: Dataset, filename: str, exts: typing.Tuple[str, str]):
    if os.path.exists(filename):
        os.remove(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    source_filename = filename + f"{exts[0]}"
    target_filename = filename + f"{exts[1]}"

    with open(source_filename, mode="w") as source, open(
        target_filename, mode="w"
    ) as target:
        for example in dataset.examples:
            source_sent = example.source
            target_sent = example.target
            source.write(" ".join(source_sent) + "\n")
            target.write(" ".join(target_sent) + "\n")


def split_dataset(
    dataset_prefix: str,
    exts: typing.Tuple[str, str],
    split_ratio: typing.Tuple[float, float, float],
    dest_path: str,
):
    train, valid, test = translation_dataset(dataset_prefix, exts, split_ratio)

    train_path = os.path.join(dest_path, "train")
    valid_path = os.path.join(dest_path, "valid")
    test_path = os.path.join(dest_path, "test")

    save_dataset(dataset=train, filename=train_path, exts=exts)
    save_dataset(dataset=valid, filename=valid_path, exts=exts)
    save_dataset(dataset=test, filename=test_path, exts=exts)


def main():
    """Usage examples:

        python -m src.split_dataset split \
            --dataset_prefix=/workspace/tmp/dataset_10k.bpe/train \
            --exts='.diff, .msg' \
            --split-ratio='0.8, 0.15, 0.05' \
            --dest-path=/workspace/tmp/test_dataset_10k
    """
    arguments = docopt(__doc__, version="Split dataset 1.0")

    if arguments["split"]:
        dataset_prefix = arguments["--dataset_prefix"]
        exts = [arg.strip() for arg in arguments["--exts"].split(",")]
        split_ratio = [
            float(arg.strip()) for arg in arguments["--split-ratio"].split(",")
        ]
        dest_path = arguments["--dest-path"]

        split_dataset(dataset_prefix, exts, split_ratio, dest_path)


if __name__ == "__main__":
    main()
