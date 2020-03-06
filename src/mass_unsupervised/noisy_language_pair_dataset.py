from typing import Dict, List, Optional, Tuple

import torch
from fairseq.data import FairseqDataset, data_utils

from .masked_dictionary import MaskedDictionary

import numpy as np


class NoisyLanguagePairDataset(FairseqDataset):
    def __init__(
        self,
        source_dataset: FairseqDataset,
        source_sizes: torch.LongTensor,
        target_dataset: Optional[FairseqDataset],
        target_sizes: Optional[torch.LongTensor],
        source_dict: MaskedDictionary,
        target_dict: Optional[MaskedDictionary],
        source_lang_id: int,
        target_lang_id: Optional[int],
        left_pad_source: bool = False,
        left_pad_target: Optional[bool] = False,
        max_source_positions: int = 1024,
        max_target_positions: Optional[int] = 1024,
        shuffle: bool = True,
        input_feeding: bool = True,
        ratio: float = 0.50,
        pred_probs: Optional[torch.FloatTensor] = None,
    ) -> None:
        self.source_dataset = source_dataset
        self.source_sizes = source_sizes
        self.target_dataset = target_dataset
        self.target_sizes = target_sizes
        self.source_dict = source_dict
        self.target_dict = target_dict
        self.source_lang_id = source_lang_id
        self.target_lang_id = target_lang_id
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.ratio = ratio
        self.pred_probs = pred_probs

    def __getitem__(self, index: int) -> Dict:
        source_item: torch.LongTensor = self.source_dataset[index]
        target_item: Optional[torch.LongTensor] = self.target_dataset[
            index
        ] if self.target_dataset else None
        source_list: List[int] = source_item.tolist()
        source: List[int] = []

        for idx, word in enumerate(source_list):
            keep = torch.rand(1).item()
            if all([idx > 0, idx < len(source_list) - 1, keep <= self.ratio]):
                source.append(self.source_dict.mask_idx)
            else:
                source.append(word)

        return {
            "id": index,
            "source": torch.LongTensor(source),
            "target": target_item,
        }

    def __len__(self) -> int:
        return len(self.source_dataset)

    def collater(self, samples: List[Dict]) -> Dict:
        return self.collate(
            samples,
            self.source_dict.pad_idx,
            self.source_dict.eos_idx,
            self.left_pad_source,
            self.left_pad_target,
            self.input_feeding,
        )

    def collate(
        self,
        samples: List[Dict],
        pad_idx: int = 3,
        eos_idx: int = 1,
        left_pad_source: bool = True,
        left_pad_target: bool = False,
        input_feeding: bool = True,
    ) -> Dict:
        if len(samples) == 0:
            return {}

        def merge(
            key: str, left_pad: bool, move_eos_to_beginning: bool = False
        ) -> torch.LongTensor:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad,
                move_eos_to_beginning,
            )

        # sort src_tokens by descending source length
        src_tokens = merge("source", left_pad=left_pad_source)
        src_lengths = torch.LongTensor([s["source"].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        src_tokens = src_tokens.index_select(0, sort_order)

        # sort ids by descending source length
        idx = torch.LongTensor([s["id"] for s in samples])
        idx = idx.index_select(0, sort_order)

        target_tokens = None
        prev_output_tokens = None

        if samples[0].get("target", None) is not None:
            target_tokens = merge("target", left_pad=left_pad_target)
            target_tokens = target_tokens.index_select(0, sort_order)
            ntokens = sum(s["target"].numel() for s in samples)

            if input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens = merge(
                    key="target",
                    left_pad=left_pad_target,
                    move_eos_to_beginning=True,
                )

                prev_output_tokens = prev_output_tokens.index_select(
                    0, sort_order
                )
        else:
            ntokens = sum(s["source"].numel() for s in samples)

        batch = {
            "id": idx,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
            },
            "target": target_tokens,
        }

        if prev_output_tokens is not None:
            batch["net_input"]["prev_output_tokens"] = prev_output_tokens

        return batch

    def num_tokens(self, index: int) -> int:
        source_size: int = self.source_sizes[index]
        target_size: int = self.target_sizes[index] if self.target_sizes else 0
        return max(source_size, target_size)

    def size(self, index: int) -> Tuple[int, int]:
        source_size: int = self.source_sizes[index]
        target_size: int = self.target_sizes[index] if self.target_sizes else 0
        return source_size, target_size

    def ordered_indices(self) -> List[int]:
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        if self.target_sizes is not None:
            indices = indices[np.argsort(self.target_sizes[indices])]

        ordered_sizes = self.source_sizes[indices]
        return indices[np.argsort(ordered_sizes)]

    @property
    def supports_prefetch(self) -> bool:
        source_support = getattr(
            self.source_dataset, "supports_prefetch", False
        )
        if self.target_dataset:
            target_support = getattr(
                self.target_dataset, "supports_prefetch", False
            )
        else:
            target_support = True

        return source_support and target_support

    def prefetch(self, indices: List[int]) -> None:
        self.source_dataset.prefetch(indices)
        if self.target_dataset:
            self.target_dataset.prefetch(indices)
