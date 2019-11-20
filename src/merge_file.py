import argparse
import os
import typing
from pathlib import Path

import orjson
from tqdm import tqdm

from src.utils import iterate_lines


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
