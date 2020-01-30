import os
import subprocess
import typing


def iterate_lines(file_path: str) -> typing.Generator[str, None, None]:
    with open(file_path, mode="r") as ptr:
        while True:
            line = ptr.readline()
            if not line:
                return
            yield line


def parse_listed_arg(arg: str, separator: str = ",") -> typing.List[typing.Any]:
    tokens = [token.strip() for token in arg.split(separator)]
    return tokens


def lines_in_file(filepath: str) -> int:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    output = subprocess.check_output(f"wc -l < {filepath}", shell=True)
    n_lines = int(output.strip())
    return n_lines


if __name__ == "__main__":
    pass
