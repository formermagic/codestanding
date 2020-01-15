"""Remove duplicates from the reference file
and picks corresponding lines from the aligned file.

Usage:
    remove_duplicates \
        --reference-filepath=<ref> \
        --aligned-filepath=<aligned> \
        --destination-path=<dst>

Options:
    --reference-filepath=<ref>      A path to the source file to filter.
    --aligned-filepath=<aligned>    A path to the target file to filter.
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
    messages_path: str,
    diffs_path: str,
    messages_output_path: str,
    diffs_output_path: str,
) -> None:
    base_dir = os.path.dirname(messages_output_path)
    pack_tmp_filepath = os.path.join(base_dir, "messages_pack_tmp.jsonl")
    sort_tmp_filepath = os.path.join(base_dir, "messages_sort_tmp.jsonl")
    no_dup_tmp_filepath = os.path.join(base_dir, "messages_no_dup_tmp.jsonl")
    sort_idx_tmp_filepath = os.path.join(
        base_dir, "messages_sort_idx_tmp.jsonl"
    )

    pack_file_content(filepath=messages_path, output_path=pack_tmp_filepath)

    sort_file(filepath=pack_tmp_filepath, output_path=sort_tmp_filepath)

    remove_duplicates(
        filepath=sort_tmp_filepath, output_path=no_dup_tmp_filepath
    )

    sort_file_by_index(
        filepath=no_dup_tmp_filepath, output_path=sort_idx_tmp_filepath
    )

    _pick_diffs(
        index_path=sort_idx_tmp_filepath,
        diffs_path=diffs_path,
        messages_output_path=messages_output_path,
        diffs_output_path=diffs_output_path,
    )

    remove_file(pack_tmp_filepath)
    remove_file(sort_tmp_filepath)
    remove_file(no_dup_tmp_filepath)
    remove_file(sort_idx_tmp_filepath)


def main():
    messages_path = "/workspace/tmp/messages.all"
    diffs_path = "/workspace/tmp/diffs.all"
    messages_output_path = "/workspace/tmp/messages_no_dup.all"
    diffs_output_path = "/workspace/tmp/diffs_no_dup.all"

    process_dataset_files(
        messages_path=messages_path,
        diffs_path=diffs_path,
        messages_output_path=messages_output_path,
        diffs_output_path=diffs_output_path,
    )


if __name__ == "__main__":
    main()
