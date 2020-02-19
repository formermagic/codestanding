from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils

from .masked_dictionary import MaskedDictionary


class MaskedLanguagePairDataset(FairseqDataset):
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
        left_pad_source: bool,
        left_pad_target: Optional[bool],
        max_source_positions: int = 1024,
        max_target_positions: Optional[int] = 1024,
        mask_prob: float = 0.15,
        block_size: int = 64,
        shuffle: bool = True,
        ratio: Optional[float] = None,
        training: bool = True,
        pred_probs: Optional[torch.FloatTensor] = None,  # [0.8, 0.1, 0.1]
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
        self.mask_prob = mask_prob
        self.block_size = block_size
        self.shuffle = shuffle
        self.ratio = ratio
        self.pred_probs = pred_probs

    def __getitem__(self, index: int) -> Dict:
        source_item: torch.LongTensor = self.source_dataset[index]
        target_item: Optional[torch.LongTensor] = None
        if self.target_dataset:
            target_item = self.target_dataset[index]

        # get masked indices with the given source item
        positions = torch.arange(0, len(source_item))
        masked_idx: List[int] = []
        for idx in range(1, len(source_item), self.block_size):
            block = positions[idx : idx + self.block_size]
            masked_len = int(len(block) * self.mask_prob)
            masked_start = np.random.choice(
                block[: len(block) - masked_len + 1], 1
            ).item()
            masked_idx.extend(
                positions[masked_start : masked_start + masked_len]
            )
        masked_indices = torch.LongTensor(masked_idx)

        # given `x1 x2 _ _ _ x6 x7` predict masked `x3 x4 x5` (target)
        # the decoder's previous output: `x2 x3 x4`
        # the decoder's current  output: `x4 x5 x6`
        target = source_item[masked_indices].clone()
        prev_output_tokens = source_item[masked_indices - 1].clone()
        positions = masked_indices + 1  # self.source_dict.pad_index=1
        source_item[masked_indices] = self.replace(source_item[masked_indices])

        samples = {
            "id": index,
            "source": source_item,
            "target": target,
            "prev_output_tokens": prev_output_tokens,
            "positions": positions,
        }

        return samples

    def replace(self, x: torch.LongTensor) -> torch.LongTensor:
        if len(x) == 0:
            return x

        # tensor with <mask> tokens
        mask_word = x.clone().fill_(self.source_dict.mask_index)
        # tensor with the real word tokens
        real_word = x.clone()
        # tensor with random words (other than specials)
        rand_word = x.clone().random_(
            self.source_dict.nspecial, len(self.source_dict)
        )

        # replacement mask for pred_probs
        probs = torch.multinomial(self.pred_probs, len(x), replacement=True)

        # applied replacement mask
        result = (
            mask_word * (probs == 0).long()
            + real_word * (probs == 1).long()
            + rand_word * (probs == 2).long()
        )

        return result

    def __len__(self) -> int:
        source_len = len(self.source_dataset)
        target_len = len(self.target_dataset) if self.target_dataset else 0
        return max(source_len, target_len)

    def collater(self, samples: List[Dict]) -> Dict:
        return self.collate(
            samples,
            pad_idx=self.source_dict.pad(),
            eos_idx=self.source_dict.eos(),
            left_pad_source=False,
            left_pad_target=False,
            input_feeding=True,
        )

    def collate(
        self,
        samples: List[Dict],
        pad_idx: int,
        eos_idx: int,
        left_pad_source: bool = True,
        left_pad_target: bool = False,
        input_feeding: bool = True,
    ) -> Dict:
        if not samples:
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
        source_tokens = merge("source", left_pad=left_pad_source)
        source_lengths = torch.LongTensor(
            [s["source"].numel() for s in samples]
        )
        source_lengths, sort_order = source_lengths.sort(descending=True)
        source_tokens = source_tokens.index_select(0, sort_order)

        # sort ids by descending source length
        idx = torch.LongTensor([s["id"] for s in samples])
        idx = idx.index_select(0, sort_order)

        # prepare previous output tokens and sort them by source length descending
        prev_output_tokens = None
        has_prev_tokens = samples[0].get("prev_output_tokens", None) is not None
        if input_feeding and has_prev_tokens:
            prev_output_tokens = merge(
                "prev_output_tokens", left_pad=left_pad_target
            )

            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        # prepare positions to predict and sort them by source length descending
        if samples[0].get("positions", None) is not None:
            positions = merge("positions", left_pad=left_pad_target)
            positions = positions.index_select(0, sort_order)
        else:
            positions = None

        # prepare masked target tokens and sort them by source length descending
        if samples[0].get("target", None) is not None:
            target = merge("target", left_pad=left_pad_target)
            target = target.index_select(0, sort_order)
            n_tokens = target.numel()
        else:
            target = None
            n_tokens = 0

        batch = {
            "id": idx,
            "nsentences": len(samples),
            "net_input": {
                "src_lengths": source_lengths,
                "src_tokens": source_tokens,
                "prev_output_tokens": prev_output_tokens,
                "positions": positions,
            },
            "target": target,
            "ntokens": n_tokens,
        }

        return batch

    def num_tokens(self, index: int) -> int:
        source_size = self.source_sizes[index]
        target_size = (
            self.target_sizes[index] if self.target_sizes is not None else 0
        )
        return max(source_size, target_size)

    def size(self, index: int) -> Tuple[int, int]:
        source_size = self.source_sizes[index]
        target_size = (
            self.target_sizes[index] if self.target_sizes is not None else 0
        )
        return source_size, target_size

    def ordered_indices(self) -> List[int]:
        if self.shuffle:
            indices = torch.randperm(len(self))
        else:
            indices = torch.arange(len(self))

        if self.target_sizes:
            indices = indices[torch.argsort(self.target_sizes[indices])]

        return indices[torch.argsort(self.source_sizes[indices])]

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
