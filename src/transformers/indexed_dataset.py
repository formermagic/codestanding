import os
import struct
from functools import lru_cache
from typing import IO, Any, AnyStr, Iterable, List, Optional, Text, Tuple, Union

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .utils import lines_in_file

_BufferType = Union[bytes, bytearray, memoryview]


def read_in_chunks(
    file_object: IO[AnyStr], chunk_size: int = 1024
) -> Iterable[AnyStr]:
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def lookahead(iterable: Iterable) -> Iterable[Tuple[Any, bool]]:
    # get an iterator and pull the first value
    iterator = iter(iterable)
    last_item = next(iterator)  # pylint: disable=stop-iteration-return
    # run the iterator to exhaustion (starting from the second value)
    for val in iterator:
        # report the *previous* value (more to come)
        yield (last_item, True)
        last_item = val
    # report the last value
    yield (last_item, False)


class IndexProcessorMixin:
    def __init__(self, vocab_size: Optional[int] = None) -> None:
        self.vocab_size = vocab_size

    @property
    def dtype(self) -> np.dtype:
        # pylint: disable=no-else-return
        if self.vocab_size is None:
            return np.dtype(np.uint16)  # type: ignore
        elif self.vocab_size < 65500:
            return np.dtype(np.uint16)  # type: ignore
        else:
            return np.dtype(np.int32)  # type: ignore

    def dtype_format(self, size: int) -> Text:
        return f"<{size}{self.dtype.char}"

    def unpack(self, buffer: _BufferType) -> List[int]:
        dtype_size = self.dtype.itemsize
        dtype_fmt = self.dtype_format(len(buffer) // dtype_size)
        unpacked_value = struct.unpack(dtype_fmt, buffer)
        return list(unpacked_value)

    def pack(self, data: List[int]) -> bytes:
        dtype_fmt = self.dtype_format(len(data))
        return struct.pack(dtype_fmt, *data)

    @lru_cache(maxsize=10_000)
    def read_tokens(self, filepath: Text, index: int) -> List[int]:
        seek = 0
        current = 0
        tokens = []
        with open(filepath, mode="rb") as _file:
            while True:
                _file.seek(seek)
                dtype_size = self.dtype.itemsize
                values = self.unpack(_file.read(dtype_size))
                if not values:
                    break

                size = values[0]
                if current == index:
                    buffer = _file.read(dtype_size * size)
                    tokens.extend(self.unpack(buffer))
                    break

                seek += (size + 1) * 2
                current += 1

        return tokens

    @lru_cache(maxsize=10_000)
    def read_lengths(self, filepath: Text) -> List[int]:
        seek = 0
        result = []
        with open(filepath, mode="rb") as _file:
            while True:
                _file.seek(seek)
                byte = _file.read(self.dtype.itemsize)
                if not byte:
                    return result
                size = self.unpack(byte)[0]
                result.append(size)
                seek += (size + 1) * 2


class IndexDatasetPreprocessor(IndexProcessorMixin):
    def __init__(
        self, filepath: Text, tokenizer: PreTrainedTokenizer, max_length: int
    ) -> None:
        super().__init__(tokenizer.vocab_size)
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess(self, output_path: Text) -> None:
        filename = os.path.basename(self.filepath).split(".")[0]
        output_filepath = os.path.join(output_path, f"{filename}.bin")
        input_file = open(self.filepath, mode="r", encoding="utf-8")
        output_file = open(output_filepath, mode="wb")

        def flush(bucket: List[Text], file: IO[AnyStr]) -> None:
            batch_encoding = self.tokenizer.batch_encode_plus(
                bucket, add_special_tokens=True, max_length=self.max_length
            )
            batch_ids = batch_encoding.data["input_ids"]

            for tokens in batch_ids:
                _tokens = [len(tokens)] + tokens
                _bytes = self.pack(_tokens)
                file.write(_bytes)

        num_lines = lines_in_file(self.filepath)
        kwargs = {"total": num_lines, "desc": "Dataset Preprocessing"}

        with input_file, output_file:
            bucket = []
            for line in tqdm(input_file, **kwargs):
                line = line.rstrip()
                bucket.append(line)
                if len(bucket) == 2 ** 16:
                    flush(bucket, output_file)
                    bucket = []
            if bucket:
                flush(bucket, output_file)

