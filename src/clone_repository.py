import argparse
import asyncio
import logging
import os
import subprocess
import typing
from concurrent.futures.process import ProcessPoolExecutor

from git import Repo
from tqdm import tqdm


def convert_url_to_name(url: str) -> str:
    name = url.replace(".git", "")
    name = name.split("/")[-2:]
    name = "@".join(name)
    return name


def clone_repository(url: str, output_path: str) -> None:
    logging.info("started cloning repo with url: %s", url)
    name = convert_url_to_name(url)
    path = os.path.join(output_path, name)
    _ = Repo.clone_from(url, path)
    logging.info("cloned %s into %s", name, path)


async def run_clone_repositories(
    repositories: typing.List[str],
    output_directory: str,
    clear_before: bool = True,
) -> None:
    if clear_before:
        subprocess.call(f"rm -rf {output_directory}", shell=True)

    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [
            loop.run_in_executor(
                executor, clone_repository, repository, output_directory
            )
            for repository in repositories
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


async def main() -> None:
    log_dir = "/workspace/logs/"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, "clone_repository.log"),
        filemode="w+",
        format="%(asctime)s %(pathname)-12s: %(message)s",
        level=logging.INFO,
    )

    logging.info("started preparing for cloning")

    parser = argparse.ArgumentParser(
        description="Clone git repositories into local directory."
    )

    parser.add_argument(
        "--repo_file",
        type=str,
        required=True,
        help="A path to the file containing the list of repositories to clone.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="A path to save cloned repositories to.",
    )
    parser.add_argument(
        "--clear_before",
        type=bool,
        required=False,
        default=True,
        help="Indicates if the output directory should be cleaned up before cloning.",
    )

    args = parser.parse_args()
    repositories_filename: str = args.repo_file
    output_directory: str = args.output
    clear_before: bool = args.clear_before

    with open(repositories_filename, "r") as file:
        repositories = [line.strip() for line in file.readlines()]
        logging.info("found %s repositories to clone", len(repositories))

    logging.info("clonning started, will clear the output: %r", clear_before)
    await run_clone_repositories(repositories, output_directory, clear_before)
    logging.info("finished cloning repositories")


if __name__ == "__main__":
    asyncio.run(main())
