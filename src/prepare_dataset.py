import argparse
import os
import re
import string
import typing
from functools import reduce

import orjson
from tqdm import tqdm

from utils import iterate_lines

JSONType = typing.Dict[str, typing.Any]


FILE_TOKEN = "<file>"
CHUNK_TOKEN = "<chunk>"
NEW_LINE_TOKEN = "<nl>"
ADD_TOKEN = "<add>"
DEL_TOKEN = "<del>"
URL_TOKEN = "<url>"
NUMBER_TOKEN = "<num>"
REF_TOKEN = "<ref>"
SHA_TOKEN = "<sha>"


def is_valid_message(line: str) -> bool:
    conditions = [
        re.match(r"\w+", line) != None,
        re.match(r"^merge", line, re.IGNORECASE) == None,
        re.match(r"^bump", line, re.IGNORECASE) == None,
    ]

    return reduce(lambda x, y: x and y, conditions)


def preprocess_message(text: str) -> str:
    text = text.lower()
    text = text.replace("\r", "\n")
    text = text.split("\n")[0]
    text = text.strip()
    text = text.translate(text.maketrans("", "", string.punctuation))
    text = re.sub(r"#\d+", REF_TOKEN, text)
    text = re.sub(r"[a-f0-9]{40,128}", SHA_TOKEN, text)
    text = re.sub(r"(?<!\.)\d+(?!\.)", NUMBER_TOKEN, text)
    return text


def preprocess_diff(text: str) -> str:
    text = text.replace("\n", f" {NEW_LINE_TOKEN} ")
    text = text.replace("\r", f" {NEW_LINE_TOKEN} ")
    text = text.replace(f"{NEW_LINE_TOKEN} +", f"{NEW_LINE_TOKEN} {ADD_TOKEN} ")
    text = text.replace(f"{NEW_LINE_TOKEN} -", f"{NEW_LINE_TOKEN} {DEL_TOKEN} ")
    text = re.sub(r"\@@.*?\@@", CHUNK_TOKEN, text)
    text = re.sub(
        r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
        URL_TOKEN,
        text,
    )
    return text.lower()


def merge_modifications(modifications: typing.List[JSONType]) -> str:
    merged_diffs: str = ""

    for modification in modifications:
        filename = modification["filename"]
        diff = modification["diff"]
        if not diff:
            continue
        merged_diffs += f"{FILE_TOKEN} {filename} {diff}"

    return merged_diffs


def split_parsed_diffs(
    input_path: str, message_path: str, diff_path: str
) -> None:
    buffer: typing.List[typing.Tuple[str, str]] = []
    max_len: int = 1000
    kwargs = {
        "total": 3_346_500,
        "unit": "it",
        "unit_scale": True,
        "leave": True,
    }

    with open(message_path, mode="w") as message_ptr, open(
        diff_path, mode="w"
    ) as diff_ptr:

        def flush_buffer():
            for (message, diff) in buffer:
                message_ptr.write(message + "\n")
                diff_ptr.write(diff + "\n")
            buffer.clear()

        for line in tqdm(iterate_lines(input_path), **kwargs):
            try:
                json: JSONType = orjson.loads(line)
            except ValueError:
                continue

            message: str = json["message"]
            modifications = json["modifications"]

            if None in [message, modifications]:
                continue
            if not is_valid_message(message):
                continue
            if len(message) > 70:
                continue

            message: str = preprocess_message(message)
            diff: str = merge_modifications(modifications)
            diff: str = preprocess_diff(diff)

            if len(message.split(" ")) < 4:
                continue
            if not diff or len(diff.split(" ")) > 1000:
                continue

            buffer.append((message, diff))

            if len(buffer) >= max_len:
                flush_buffer()
        flush_buffer()


def main():
    parser = argparse.ArgumentParser(
        description="Filter parsed commits by message and merged diff across files"
        "and then split into two files for messages and diffs."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="A path to the parsed diffs to filter and split.",
    )
    parser.add_argument(
        "--message_path",
        type=str,
        required=True,
        help="A path to the prepared messages.",
    )
    parser.add_argument(
        "--diff_path",
        type=str,
        required=True,
        help="A path to the prepared diffs.",
    )

    args = parser.parse_args()
    input_path = args.input_path
    message_path = args.message_path
    diff_path = args.diff_path

    if os.path.exists(message_path):
        os.remove(message_path)
    if os.path.exists(diff_path):
        os.remove(diff_path)

    split_parsed_diffs(input_path, message_path, diff_path)


if __name__ == "__main__":
    main()
