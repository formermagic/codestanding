import argparse
import asyncio
import logging
import os
import re
import typing
from abc import ABC
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import jsonlines
import nltk
from tqdm import tqdm


@dataclass
class ModificationObj:
    added: int
    removed: int
    change_type: str
    diff: str
    filename: str
    old_path: str
    new_path: str
    diff_tokens: typing.Optional[typing.List[str]] = None


@dataclass
class CommitObj:
    sha: str
    repository: str
    message: str
    modifications: typing.List[ModificationObj]

    def to_json(self) -> typing.Dict[str, typing.Any]:
        kwargs = self.__dict__
        kwargs["modifications"] = [mod.__dict__ for mod in self.modifications]
        return kwargs


class PatchTokenizer(ABC):
    def tokenize(
        self, patch: str, kwargs: typing.Dict[str, typing.Any]
    ) -> typing.List[str]:
        pass


class NaivePatchTokenizer(PatchTokenizer):
    def __init__(self):
        self.punctuation = r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        self.addition_start_token = "ADDS"
        self.deletion_start_token = "DELS"
        self.addition_end_token = "ADDE"
        self.deletion_end_token = "DELE"

        self.mwe_tokenizer = nltk.MWETokenizer(separator="")
        self.mwe_tokenizer.add_mwe(("<", self.addition_start_token, ">"))
        self.mwe_tokenizer.add_mwe(("<", self.deletion_start_token, ">"))
        self.mwe_tokenizer.add_mwe(("<", self.addition_end_token, ">"))
        self.mwe_tokenizer.add_mwe(("<", self.deletion_end_token, ">"))

    def tokenize(
        self, patch: str, kwargs: typing.Dict[str, typing.Any]
    ) -> typing.List[str]:
        ignore_punctuation: bool = kwargs.get("ignore_punctuation", False)
        prepared_string = (
            patch.lower()
            .replace("\n+", f"\n<{self.addition_start_token}>")
            .replace("\n-", f"\n<{self.deletion_start_token}>")
        )
        prepared_string = re.sub(r"\@@.*?\@@", "", prepared_string)
        prepared_string = re.sub(
            f"([{self.punctuation}])", r" \1 ", prepared_string
        )

        lines = prepared_string.split("\n")
        lines = [self.__prepare_line(line) for line in lines]
        tokens = nltk.word_tokenize("\n".join(lines))
        tokens = self.mwe_tokenizer.tokenize(tokens)

        if ignore_punctuation:
            tokens = [
                token for token in tokens if token not in self.punctuation
            ]

        return tokens

    def __prepare_line(self, line: str) -> str:
        if line.startswith(f" < {self.addition_start_token} > "):
            return line + f" < {self.addition_end_token} > "
        if line.startswith(f" < {self.deletion_start_token} > "):
            return line + f" < {self.deletion_end_token} > "
        return line


class CommitBucket:
    def __init__(self, repository_path: str):
        self.repository_path = repository_path
        self.filename = os.path.basename(repository_path)

    def iterate_commits(self) -> typing.Generator[CommitObj, None, None]:
        with jsonlines.open(self.repository_path) as reader:
            reader: jsonlines.Reader = reader
            # pylint: disable=no-member
            for obj in reader.iter(type=dict):
                commit = CommitObj(**obj)
                commit.modifications = [
                    ModificationObj(**mod) for mod in obj["modifications"]
                ]

                yield commit


class CommitBucketTokenizer:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

    async def tokenize_commit_buckets(
        self, buckets: typing.List[CommitBucket], output_path: str
    ) -> None:
        os.makedirs(output_path, exist_ok=True)
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                loop.run_in_executor(
                    executor, self._tokenize_commit_bucket, bucker, output_path
                )
                for bucker in buckets
            ]

            kwargs = {
                "total": len(futures),
                "unit": "it",
                "unit_scale": True,
                "leave": True,
            }

            with tqdm(**kwargs) as pbar:
                for future in futures:
                    future.add_done_callback(lambda _: pbar.update(1))
                await asyncio.gather(*futures, return_exceptions=True)

    def _tokenize_commit_bucket(
        self, bucket: CommitBucket, output_path: str
    ) -> None:
        # os.makedirs(output_path, exist_ok=True)

        tokenizer = NaivePatchTokenizer()
        kwargs = {"ignore_punctuation": False}
        buffer: typing.List[CommitObj] = []
        max_len = 100

        def flush(writer: jsonlines.Writer):
            writer.write_all([obj.to_json() for obj in buffer])
            buffer.clear()

        filename = os.path.join(output_path, bucket.filename)
        with jsonlines.open(filename, mode="w") as writer:
            writer: jsonlines.Writer = writer

            for commit in bucket.iterate_commits():
                for modification in commit.modifications:
                    modification.diff_tokens = tokenizer.tokenize(
                        modification.diff, kwargs
                    )

                buffer.append(commit)
                if len(buffer) > max_len:
                    flush(writer)
            flush(writer)


async def main():
    log_dir = "/workspace/logs/"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "tokenize_diffs.log"),
        filemode="w+",
        format="%(asctime)s %(pathname)-12s: %(message)s",
        level=logging.INFO,
    )

    logging.info("started preparing for tokenizing diffs")

    parser = argparse.ArgumentParser(
        description="Tokenize diffs from parsed commits."
    )

    parser.add_argument(
        "--diff_dir",
        type=str,
        required=True,
        help="A path to the parsed diffs to tokenize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="A path to write resulting tokenized diffs to.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        required=True,
        default=20,
        help="A max number of workers to tokenize commit diffs.",
    )

    nltk.download("punkt")

    args = parser.parse_args()
    input_path = args.diff_dir  # "tmp"
    output_path = args.output  # "tmp/output"
    max_workers = args.max_workers  # 20

    tokenizer = CommitBucketTokenizer(max_workers)
    buckets = [
        CommitBucket(str(path))
        for path in list(Path(input_path).glob("*.jsonl"))
    ]

    await tokenizer.tokenize_commit_buckets(buckets, output_path)


if __name__ == "__main__":
    asyncio.run(main())
