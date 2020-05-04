from typing import List, Optional, Text

from tokenizers import AddedToken, ByteLevelBPETokenizer

from transformers.tokenization_roberta import RobertaProcessing
from transformers.tokenization_utils import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "merges_file": {},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


class CodeBertTokenizerFast(PreTrainedTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file: Text,
        merges_file: Text,
        bos_token: Text = "<s>",
        eos_token: Text = "</s>",
        sep_token: Text = "</s>",
        cls_token: Text = "<s>",
        unk_token: Text = "<unk>",
        pad_token: Text = "<pad>",
        mask_token: Text = "<mask>",
        add_prefix_space: bool = True,
        trim_offsets: bool = True,
        **kwargs
    ) -> None:
        kwargs.setdefault("pad_token", pad_token)
        kwargs.setdefault("sep_token", sep_token)
        kwargs.setdefault("cls_token", cls_token)
        kwargs.setdefault("mask_token", mask_token)

        byte_tokenizer = ByteLevelBPETokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            add_prefix_space=add_prefix_space,
            lowercase=True,
        )

        super().__init__(
            byte_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs
        )

        backend_tokenizer = byte_tokenizer._tokenizer
        backend_tokenizer.post_processor = RobertaProcessing(
            sep=(sep_token, self.sep_token_id),
            cls=(cls_token, self.cls_token_id),
            trim_offsets=trim_offsets,
            add_prefix_space=add_prefix_space,
        )

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1
                    if x in [self.sep_token_id, self.cls_token_id]
                    else 0,
                    token_ids_0,
                )
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return (
            [1]
            + ([0] * len(token_ids_0))
            + [1, 1]
            + ([0] * len(token_ids_1))
            + [1]
        )

    @PreTrainedTokenizer.mask_token.setter
    def mask_token(self, value):
        if not isinstance(value, AddedToken):
            value = AddedToken(value, lstrip=True)

        self._mask_token = str(value)
        self._maybe_update_backend([value])

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
