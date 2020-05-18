import logging
from typing import List, Optional, Text, Union

from tokenizers import AddedToken
from transformers import RobertaTokenizerFast
from transformers.tokenization_roberta import RobertaProcessing

logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "merges_file": {},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}

# pylint: disable=abstract-method
class CodeBertTokenizerFast(RobertaTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # pylint: disable=too-many-arguments
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

        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )

        self.backend_tokenizer._tokenizer.post_processor = RobertaProcessing(
            sep=(sep_token, self.sep_token_id),
            cls=(cls_token, self.cls_token_id),
            trim_offsets=trim_offsets,
            add_prefix_space=add_prefix_space,
        )

        self.backend_tokenizer.add_special_tokens([kwargs["mask_token"]])

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

            def decode_idx(value: int) -> int:
                if value in [self.sep_token_id, self.cls_token_id]:
                    return 1
                return 0

            return [decode_idx(idx) for idx in token_ids_0]

        first_seq = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is None:
            return first_seq

        second_seq = [1] + ([0] * len(token_ids_1)) + [1]
        return first_seq + second_seq

    @property
    def mask_token(self) -> Text:
        # pylint: disable=arguments-differ
        if self._mask_token is None:
            logger.error("Using mask_token, but it is not set yet.")
        return self._mask_token

    @mask_token.setter
    def mask_token(self, value: Union[Text, AddedToken]) -> None:
        if not isinstance(value, AddedToken):
            value = AddedToken(value, lstrip=True)

        self._mask_token = str(value)
        self._maybe_update_backend([value])

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
