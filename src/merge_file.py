"""Util methods to merge multiple files.

Usage:
    merge_file merge-jsonl \
        --input-path=<inp> \
        --output-path=<out_file> \
        [--remove-files]
    merge_file merge-pairs \
        --input-path=<inp> \
        --output-prefix=<out_pref> \
        --extensions=<exts> \
        [--remove-files]

Options:
    --input-path=<inp>              A path to look files at.
    --output-path=<out_file>        A path to the file to write the merged data to.
    --output-prefix=<out_pref>      A common prefix for merged files to write to.
    --extensions=<exts>             A string with extensions of paired files (a tuple of <s1, s2>).
    --remove-files                  Indicates whether input files \
                                    will be removed after completion [default: False].
"""
import os
import typing
from pathlib import Path

import orjson
from docopt import docopt
from tqdm import tqdm

from src.utils import iterate_lines, parse_listed_arg


def merge_jsonl_files(
    input_dir: str, output_filename: str, remove_files: bool = False
) -> None:
    if os.path.exists(output_filename):
        os.remove(output_filename)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    files: typing.List[Path] = list(Path(input_dir).glob("*.jsonl"))
    kwargs = {
        "total": len(files),
        "unit": "it",
        "unit_scale": True,
        "leave": True,
    }

    with open(output_filename, "a") as output:
        for file in tqdm(files, **kwargs):
            file_path = file.as_posix()

            if file_path == output_filename:
                continue

            for line in iterate_lines(file_path):
                output.write(line)

            if remove_files:
                os.remove(file_path)


def merge_pair_files(
    input_path: str,
    output_prefix: str,
    extensions: typing.Tuple[str, str],
    remove_files: bool = True,
) -> None:
    dirname = os.path.dirname(output_prefix)
    os.makedirs(dirname, exist_ok=True)

    search_path = Path(input_path)
    source, target = extensions
    source_file = open(output_prefix + f".{source}", mode="w")
    target_file = open(output_prefix + f".{target}", mode="w")

    with source_file, target_file:
        for source_path in search_path.glob(f"**/*.{source}"):
            prefix_path = os.path.splitext(source_path)[0]
            target_path = prefix_path + f".{target}"
            if not os.path.exists(target_path):
                print(target_path)
                continue

            print(f"source: {source_path}")
            with open(source_path) as file:
                source_file.writelines(file.readlines())

            print(f"target: {target_path}")
            with open(target_path) as file:
                target_file.writelines(file.readlines())


def main():
    parser = argparse.ArgumentParser(
        description="Merges parsed diff json lines into a single file."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="A path to the parsed diff files.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="A full name for the output merged file.",
    )
    parser.add_argument(
        "--remove_files",
        action="store_true",
        required=False,
        default=False,
        help="Indicates if input files should be deleted.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    output_filename = args.output_filename
    remove_files = args.remove_files
    merge_jsonl_files(input_dir, output_filename, remove_files=remove_files)


if __name__ == "__main__":
    main()
