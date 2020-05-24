import linecache
from typing import List, Text

from torch.utils.data import Dataset

from .utils import lines_in_file


class MonoTextBatchDataset(Dataset):
    def __init__(self, data_path: Text) -> None:
        self.data_path = data_path
        self.num_sentences = lines_in_file(data_path)

    def __len__(self) -> int:
        return self.num_sentences

    def getline(self, index: int) -> Text:
        return linecache.getline(self.data_path, index).rstrip()

    def __getitem__(self, indices: List[int]) -> List[Text]:
        lines = [self.getline(idx) for idx in indices]
        lines = sorted(lines, key=len, reverse=True)
        return lines
