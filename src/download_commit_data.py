"Docs... TODO"

import asyncio
import os
import pprint
import typing
from concurrent.futures.thread import ThreadPoolExecutor

import jsonlines
import yaml
from github import Commit, Github, PaginatedList, Repository
from tqdm import tqdm

CommitData = typing.Mapping[str, typing.Any]


def pprint_unsorted(x: object):
    pprint.sorted = lambda x, key=None: x
    pprint.pprint(x)


class Config:
    token: str

    def __init__(self, filename: str):
        with open(filename, mode="r") as file:
            data = yaml.safe_load(file)
            self.token = data["user"]["token"]


class CommitScraper:
    def __init__(self, repository: Repository):
        self.repository = repository


    def parse(self, commit):
        return {
                "repo": self.repository.full_name,
                "url": commit.url,
                "sha": commit.sha,
                "language": self.repository.language,
                "message": commit.commit.message,
                "files": [
                    {
                        "sha": file.sha,
                        "name": os.path.splitext(file.filename)[0],
                        "extension": os.path.splitext(file.filename)[1],
                        "blob_url": file.blob_url,
                        "raw_url": file.raw_url,
                        "patch": file.patch,
                        "status": file.status,
                        "changes": file.changes,
                        "additions": file.additions,
                        "deletions": file.deletions,
                    }
                    for file in commit.files
                ],
                "stats": {
                    "changes": commit.stats.total,
                    "additions": commit.stats.additions,
                    "deletions": commit.stats.deletions,
                },
            }

    def fetch_data(
        self, output_filename: str, show_progress_bar: bool = True
    ) -> None:
        """Docstring..."""
        # with open(output_filename, mode="w") as file, jsonlines.Writer(
        #     file
        # ) as json_writer:

        commits: PaginatedList = self.repository.get_commits()
        iterable: typing.Iterable[Commit]

        if show_progress_bar:
            progress_bar = tqdm(
                commits,
                desc=f"Fetching commit data from {self.repository.full_name}",
                total=commits.totalCount,
            )
            iterable = progress_bar
        else:
            iterable = commits

        # from functools import reduce
        # page_count = max(int(commits.totalCount / 100.), 1)
        # print(f"page_count={page_count}")
        # pages = reduce(lambda x,y: x+y, [commits.get_page(page) for page in range(page_count)])
        # commits = [self.parse(page) for page in pages]
        # print(f"pages={len(pages)}")

        for commit in commits:
            commit_data: CommitData = {
                "repo": self.repository.full_name,
                "url": commit.url,
                "sha": commit.sha,
                "language": self.repository.language,
                "message": commit.commit.message,
                "files": [
                    {
                        "sha": file.sha,
                        "name": os.path.splitext(file.filename)[0],
                        "extension": os.path.splitext(file.filename)[1],
                        "blob_url": file.blob_url,
                        "raw_url": file.raw_url,
                        "patch": file.patch,
                        "status": file.status,
                        "changes": file.changes,
                        "additions": file.additions,
                        "deletions": file.deletions,
                    }
                    for file in commit.files
                ],
                # "stats": {
                #     "changes": commit.stats.total,
                #     "additions": commit.stats.additions,
                #     "deletions": commit.stats.deletions,
                # },
            }

                # # pylint: disable=no-member
                # json_writer.write(commit_data)

    @staticmethod
    def read_data(filenane: str) -> typing.Iterable[CommitData]:
        with open(filenane, mode="r") as file, jsonlines.Reader(
            file
        ) as json_reader:
            # pylint: disable=no-member
            for obj in json_reader.iter():
                yield obj


# scraper = CommitScraper(repository=repo)
# scraper.fetch_data(output_filename="./src/test3.jsonl")
# read_data = next(CommitScraper.read_data("./src/test2.jsonl"))
# pprint_unsorted(read_data)


class GithubBuilder:
    def __init__(self):
        pass

    def build(self, config: Config) -> Github:
        g = Github(login_or_token=config.token, per_page=100)
        return g


class CommitScraperFacade:
    def __init__(self, config: Config, github_builder: GithubBuilder):
        self.config = config
        self.github_builder = github_builder

    def download_repository_data(
        self,
        repository_name: str,
        output_dir: str,
        show_progress_bar: bool = True,
    ):
        g = self.github_builder.build(self.config)
        repository = g.get_repo(repository_name)
        scraper = CommitScraper(repository=repository)
        repo_name = repository_name.replace("/", "_")
        output_filename = os.path.join(output_dir, f"{repo_name}.jsonl")

        scraper.fetch_data(
            output_filename=output_filename, show_progress_bar=show_progress_bar
        )

    def download_data(
        self,
        repositories: typing.List[str],
        output_dir: str,
        show_progress_bar: bool = True,
    ):
        if not os.path.exists(output_dir):
            os.makedirs(name=output_dir, exist_ok=True)
        for repo_name in repositories:
            self.download_repository_data(
                repository_name=repo_name,
                output_dir=output_dir,
                show_progress_bar=show_progress_bar,
            )


class CommitScraperAsyncFacade:
    def __init__(self, config: Config, github_builder: GithubBuilder):
        self.config = config
        self.github_builder = github_builder

    async def download_data_async(
        self, repositories: typing.List[str], output_dir: str
    ) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(name=output_dir, exist_ok=True)
        loop: asyncio.BaseEventLoop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                loop.run_in_executor(
                    executor, self._download_data, name, output_dir
                )
                for name in repositories
            ]
            _ = [
                await response
                for response in tqdm(
                    asyncio.as_completed(futures), total=len(repositories)
                )
            ]

    def _download_data(self, repository_name: str, output_dir: str) -> None:
        scraper = CommitScraperFacade(
            config=self.config, github_builder=self.github_builder
        )

        scraper.download_repository_data(
            repository_name=repository_name,
            output_dir=output_dir,
            show_progress_bar=False,
        )


async def main():
    config = Config(filename="./config/github.yml")
    builder = GithubBuilder()
    repositories = [
        # "ansible/ansible"
        "vlarine/transformers-ru"
    ]
    scraper = CommitScraperAsyncFacade(config=config, github_builder=builder)
    await scraper.download_data_async(
        repositories=repositories, output_dir="./tmp/data"
    )


# def main():
#     config = Config(filename="./config/github.yml")
#     repositories = ["rsennrich/subword-nmt", "vlarine/transformers-ru", "amirziai/flatten", "kaushaltrivedi/fast-bert"]
#     scraper = CommitScraperFacade(config=config)
#     scraper.download_data(repositories=repositories, output_dir="./tmp/data")

if __name__ == "__main__":
    import time

    s = time.perf_counter()
    asyncio.run(main())
    # main()
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")
