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

            if remove_files:
                os.remove(source_path)
                os.remove(target_path)


def main():
    """
    python -m src.merge_file merge-pairs \
        --input-path=/workspace/tmp/ast_test/test-01 \
        --output-prefix=/workspace/tmp/ast_test/test-02/all \
        --extensions="src, ast" \
        --remove-files

    python -m src.merge_file merge-jsonl \
        --input-path=/workspace/tmp/test-01 \
        --output-path=123 \
        --remove-files=123
    """
    arguments = docopt(__doc__, version="Merge files utils 1.0")
    print(arguments, arguments["--remove-files"])

    if arguments["merge-jsonl"]:
        input_path = str(arguments["--input-path"])
        output_path = str(arguments["--output-path"])
        remove_files = bool(arguments["--remove-files"])

        merge_jsonl_files(input_path, output_path, remove_files)

    elif arguments["merge-pairs"]:
        extensions = parse_listed_arg(str(arguments["--extensions"]))
        extensions = str(extensions[0]), str(extensions[1])
        input_path = str(arguments["--input-path"])
        output_prefix = str(arguments["--output-prefix"])
        remove_files = bool(arguments["--remove-files"])

        merge_pair_files(input_path, output_prefix, extensions, remove_files)


if __name__ == "__main__":
    main()
