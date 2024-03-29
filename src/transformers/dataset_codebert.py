import linecache
from typing import Text

import torch
from torch.utils.data import Dataset

from .tokenization_codebert import CodeBertTokenizerFast
from .utils import lines_in_file


class CodeBertDataset(Dataset):
    def __init__(
        self, tokenizer: CodeBertTokenizerFast, data_path: Text, max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.num_sentences = lines_in_file(data_path)

    def __len__(self) -> int:
        return self.num_sentences

    def __getitem__(self, index: int) -> torch.Tensor:
        line = linecache.getline(self.data_path, index).rstrip()
        batch_encoding = self.tokenizer.encode_plus(
            line, add_special_tokens=True, max_length=self.max_length
        )

        # pylint: disable=not-callable
        input_ids = torch.tensor(batch_encoding["input_ids"], dtype=torch.long)

        return input_ids
