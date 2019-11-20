import typing


def iterate_lines(file_path: str) -> typing.Generator[str, None, None]:
    with open(file_path, mode="r") as ptr:
        while True:
            line = ptr.readline()
            if not line:
                return
            yield line


if __name__ == "__main__":
    pass
