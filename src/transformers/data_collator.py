from itertools import chain
from typing import Dict, Iterator, List, Text

import torch
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
)


def chuncked_list(data: List, chunck_size: int) -> Iterator[List]:
    for i in range(0, len(data), chunck_size):
        yield data[i : i + chunck_size]


class LazyBatchDataCollator(DataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._collator = DataCollatorForLanguageModeling(tokenizer)

    # pylint: disable=arguments-differ
    def collate_batch(
        self, batch_samples: List[List[Text]]
    ) -> List[Dict[str, torch.Tensor]]:
        flatten = chain.from_iterable(batch_samples)
        flatten_sorted = sorted(flatten, key=len, reverse=True)
        chunck_size = len(batch_samples[0])
        sorted_samples = chuncked_list(flatten_sorted, chunck_size)

        batch_result = []
        for samples in sorted_samples:
            batch_encoding = self.tokenizer.batch_encode_plus(
                samples, add_special_tokens=True, max_length=self.max_length
            )

            # pylint: disable=not-callable
            batch_ids: List[List[int]] = batch_encoding.data["input_ids"]
            batch_tensors = [torch.tensor(ids) for ids in batch_ids]
            collated_samples = self._collator.collate_batch(batch_tensors)
            batch_result.append(collated_samples)
        return batch_result
