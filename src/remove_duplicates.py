"""Remove duplicates from the reference file
and picks corresponding lines from the aligned file.

Usage:
    remove_duplicates \
        --reference-filepath=<ref> \
        --aligned-filepath=<aligned> \
        --destination-path=<dst>

Options:
    --reference-filepath=<ref>      A path to the file to remove duplicates from.
    --aligned-filepath=<aligned>    A path to the file that should be aligned \
                                    to filtered reference lines.
    --destination-path=<dst>        Destination path to write results to.
"""
import os
import subprocess
import typing

from docopt import docopt

from .utils import iterate_lines


def pack_file_content(filepath: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w") as output_ptr:
        for idx, line in enumerate(iterate_lines(filepath)):
            line = line.replace("\n", "")
            output_ptr.write(f"{idx} {line}\n")
            # if idx == 10000:
            #     break


def split_indexed_line(line: str) -> typing.Tuple[int, str]:
    splited = line.split(" ")
    idx: int = int(splited[0])
    line: str = " ".join(splited[1:])
    return idx, line


def sort_file_by_index(filepath: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.call(f"sort -g -k 1 {filepath} > {output_path}", shell=True)


def sort_file(filepath: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.call(f"sort -k 2 {filepath} > {output_path}", shell=True)


def is_english(text: str) -> bool:
    try:
        text.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    return True


def remove_duplicates(filepath: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode="w") as output_ptr:
        prev_line: str = ""
        for indexed_line in iterate_lines(filepath):
            (_, line) = split_indexed_line(indexed_line)
            if line != prev_line and is_english(line):
                output_ptr.write(indexed_line)
            prev_line = line


def next_indexed_item(
    generator: typing.Generator[str, None, None]
) -> typing.Tuple[int, str]:
    try:
        indexed_line: str = next(generator)
        idx, line = split_indexed_line(indexed_line)
        return idx, line
    except StopIteration as error:
        raise error


def pick_aligned_pairs(
    indexed_reference_filepath: str,
    aligned_filepath: str,
    dest_reference_filepath: str,
    dest_aligned_filepath: str,
) -> None:
    reference_iter = iterate_lines(indexed_reference_filepath)
    reference_idx, reference_sent = next_indexed_item(reference_iter)
    reference_ptr = open(dest_reference_filepath, mode="w")
    aligned_ptr = open(dest_aligned_filepath, mode="w")

    with reference_ptr, aligned_ptr:
        for idx, sent in enumerate(iterate_lines(aligned_filepath)):
            if idx != reference_idx:
                continue

            reference_ptr.write(reference_sent)
            aligned_ptr.write(sent)

            try:
                reference_idx, reference_sent = next_indexed_item(
                    reference_iter
                )
            except StopIteration:
                break


def remove_file(filepath: str) -> None:
    if os.path.exists(filepath):
        os.remove(filepath)


def process_dataset_files(
    reference_filepath: str,
    aligned_filepath: str,
    dest_reference_filepath: str,
    dest_aligned_filepath: str,
) -> None:
    base_dir = os.path.dirname(dest_reference_filepath)
    pack_tmp_filepath = os.path.join(base_dir, "_pack_tmp.jsonl")
    sort_tmp_filepath = os.path.join(base_dir, "_sort_tmp.jsonl")
    no_dup_tmp_filepath = os.path.join(base_dir, "_no_dup_tmp.jsonl")
    sort_idx_tmp_filepath = os.path.join(base_dir, "_sort_idx_tmp.jsonl")

    # convert lines into tuples (index, line_value)
    pack_file_content(
        filepath=reference_filepath, output_path=pack_tmp_filepath
    )

    # sort lines by `line_value` alphanumerically
    sort_file(filepath=pack_tmp_filepath, output_path=sort_tmp_filepath)

    # remove sequential duplicates
    remove_duplicates(
        filepath=sort_tmp_filepath, output_path=no_dup_tmp_filepath
    )

    # sort lines by `index` in an ascending order
    sort_file_by_index(
        filepath=no_dup_tmp_filepath, output_path=sort_idx_tmp_filepath
    )

    # pick lines from aligned file that correspond
    # to the positional index of indexed reference file's lines
    pick_aligned_pairs(
        indexed_reference_filepath=sort_idx_tmp_filepath,
        aligned_filepath=aligned_filepath,
        dest_reference_filepath=dest_reference_filepath,
        dest_aligned_filepath=dest_aligned_filepath,
    )

    # remove tmp files
    remove_file(pack_tmp_filepath)
    remove_file(sort_tmp_filepath)
    remove_file(no_dup_tmp_filepath)
    remove_file(sort_idx_tmp_filepath)


def main() -> None:
    """
    Usage:
        python -m src.remove_duplicates \
            --reference-filepath=/workspace/tmp/ast_test/test-02/all.src \
            --aligned-filepath=/workspace/tmp/ast_test/test-02/all.ast \
            --destination-path=/workspace/tmp/ast_test/test-02/no_dups
    """
    arguments = docopt(__doc__, version="Remove duplicates 1.0")

    reference_filepath = arguments["--reference-filepath"]
    aligned_filepath = arguments["--aligned-filepath"]
    destination_path = arguments["--destination-path"]

    dest_reference_filepath = os.path.join(
        destination_path, os.path.basename(reference_filepath)
    )
    dest_aligned_filepath = os.path.join(
        destination_path, os.path.basename(aligned_filepath)
    )

    process_dataset_files(
        reference_filepath,
        aligned_filepath,
        dest_reference_filepath,
        dest_aligned_filepath,
    )


if __name__ == "__main__":
    main()
