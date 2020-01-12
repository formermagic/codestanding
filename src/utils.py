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


if __name__ == "__main__":
    pass
