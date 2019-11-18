import argparse
import asyncio
import logging
import os
import typing
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import jsonlines
from git.exc import (
    InvalidGitRepositoryError,
    NoSuchPathError,
    RepositoryDirtyError,
)
from pydriller import Commit, RepositoryMining
from pydriller.domain.commit import Modification
from pydriller.git_repository import GitRepository, Repo
from tqdm import tqdm

JSONType = typing.Dict[str, typing.Any]
JSONList = typing.List[JSONType]


class DiffMining:
    def __init__(self, repository_path: str):
        self.repository = GitRepository(repository_path)
        self.repository_mining = RepositoryMining(repository_path)

    def _parse_modification(self, modification: Modification) -> JSONType:
        return {
            "added": modification.added,
            "removed": modification.removed,
            "change_type": modification.change_type.name,
            "diff": modification.diff,
            "filename": modification.filename,
            "old_path": modification.old_path,
            "new_path": modification.new_path,
        }

    def _parse_commit(self, commit: Commit) -> typing.Optional[JSONType]:
        try:
            modifications = commit.modifications
        except (
            InvalidGitRepositoryError,
            NoSuchPathError,
            RepositoryDirtyError,
            AttributeError,
        ):
            modifications = []

        if not modifications:
            return None

        return {
            "sha": commit.hash,
            "repository": commit.project_name,
            "message": commit.msg,
            "modifications": [
                self._parse_modification(modification)
                for modification in modifications
            ],
        }

    # TODO: Make generator-like
    def parse_diff_tree(self) -> JSONList:
        return [
            self._parse_commit(commit)
            for commit in self.repository_mining.traverse_commits()
        ]

    def download_diff_tree(self, output_path: str) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        filename = os.path.join(
            output_path,
            os.path.basename(self.repository.project_name) + ".jsonl",
        )

        logging.info(
            "starting parsing diffs for %s repository to %s",
            self.repository.project_name,
            filename,
        )

        with open(filename, mode="w"), jsonlines.open(
            filename, mode="w"
        ) as writer:
            buffer = []

            def flush_buffer():
                # pylint: disable=no-member
                writer.write_all(buffer)
                buffer.clear()

            for commit in self.repository_mining.traverse_commits():
                diff = self._parse_commit(commit)
                if diff:
                    buffer.append(diff)
                if len(buffer) == 100:
                    flush_buffer()

            if buffer:
                flush_buffer()

        logging.info(
            "finished parsing diffs for %s repository to %s",
            self.repository.project_name,
            filename,
        )


class DiffLoader:
    def __init__(self):
        pass

    async def download(
        self, repositories: typing.List[str], output_path: str
    ) -> None:
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    self._download_repository,
                    repository_path,
                    output_path,
                )
                for repository_path in repositories
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

    def _download_repository(
        self, repository_path: str, output_path: str
    ) -> None:
        miner = DiffMining(repository_path=repository_path)
        miner.download_diff_tree(output_path=output_path)


async def main() -> None:
    log_dir = "/workspace/logs/"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "parse_diffs.log"),
        filemode="w+",
        format="%(asctime)s %(pathname)-12s: %(message)s",
        level=logging.INFO,
    )

    logging.info("started preparing for parsing diffs")

    parser = argparse.ArgumentParser(
        description="Parse and download diffs from commits in the given repositories."
    )

    parser.add_argument(
        "--repo_dir",
        type=str,
        required=True,
        help="A path to the cloned repositories to parse diffs from.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="A path to write resulting parsed diffs to.",
    )

    args = parser.parse_args()
    repositories_directory = args.repo_dir
    output_directory = args.output

    repositories = [
        str(path.resolve())
        for path in Path(repositories_directory).iterdir()
        if not path.is_file()
    ]

    logging.info(
        "found %d cloned repositories to be used to analyze", len(repositories)
    )
    loader = DiffLoader()
    await loader.download(
        repositories=repositories, output_path=output_directory
    )

    logging.info("finished parsing diffs")


if __name__ == "__main__":
    asyncio.run(main())
