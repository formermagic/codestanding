"""Clean sentence pairs from the given dataset.

Usage:
    dataset_cleaner.py clean \
        --dataset-prefix=<path> \
        --extensions=<exts> \
        --output-path=<output>

Options:
    --dataset-prefix=<path>     A common prefix for the dataset path.
    --extensions=<exts>         A comma separated string with file extensions,
                                requires 2 separated arguments (source, target).
    --output-path=<output>      An output path to write results to.
"""
import os
import typing

from docopt import docopt
from tqdm import tqdm

from .utils import iterate_lines, lines_in_file, parse_listed_arg


class TranslationDatasetCleaner:
    def __init__(
        self, dataset_prefix: str, extensions: typing.Tuple[str, str]
    ) -> None:
        self.dataset_prefix = dataset_prefix
        self.extensions = extensions
        self.source_filepath = dataset_prefix + f".{extensions[0]}"
        self.target_filepath = dataset_prefix + f".{extensions[1]}"

    def clean(self, output_path: str, max_len: int = 4096) -> None:
        source_n_lines = lines_in_file(self.source_filepath)
        target_n_lines = lines_in_file(self.target_filepath)
        assert (
            source_n_lines == target_n_lines
        ), "Number of lines must match in both files!"

        basename = os.path.basename(self.dataset_prefix)
        dest_source_filepath = os.path.join(
            output_path, basename + f".{self.extensions[0]}"
        )
        dest_target_filepath = os.path.join(
            output_path, basename + f".{self.extensions[1]}"
        )

        os.makedirs(output_path, exist_ok=True)
        dest_source_file = open(dest_source_filepath, mode="w")
        dest_target_file = open(dest_target_filepath, mode="w")

        kwargs = {
            "total": source_n_lines,
            "unit": "it",
            "unit_scale": True,
            "leave": True,
        }

        with dest_source_file, dest_target_file:
            files = zip(
                iterate_lines(self.source_filepath),
                iterate_lines(self.target_filepath),
            )

            for src, trg in tqdm(files, **kwargs):
                if len(src) > max_len:
                    continue
                dest_source_file.write(src)
                dest_target_file.write(trg)


def parse_extensions(arg: str) -> typing.Tuple[str, str]:
    listed_args = parse_listed_arg(arg)
    return listed_args[:2]


def main() -> None:
    """
    python -m src.dataset_cleaner clean \
        --dataset-prefix=/workspace/tmp/code2ast/test_dataset/all \
        --extensions="src, ast" \
        --output-path=/workspace/tmp/code2ast/test_dataset_clean
    """
    arguments = docopt(__doc__, version="Dataset Cleaner 1.0")
    dataset_prefix = str(arguments["--dataset-prefix"])
    extensions = parse_extensions(arguments["--extensions"])
    output_path = str(arguments["--output-path"])

    dataset_cleaner = TranslationDatasetCleaner(dataset_prefix, extensions)
    dataset_cleaner.clean(output_path)


if __name__ == "__main__":
    main()
