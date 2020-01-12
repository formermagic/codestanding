"""Tokenization actions for the collected dataset.

Usage:
    tokenize.py train --source-input-path=<s> \
        --source-model-name=<s> \
        [--source-vocab-size=<s>] \
        [--source-input_sentence_size=<s>] \
        --target-input-path=<t> \
        --target-model-name=<t> \
        [--target-vocab-size=<t>] \
        [--target-input_sentence_size=<t>]
    tokenize.py tokenize-bpe --task=<s> \
        --source-model=<s> \
        --source-path=<s> \
        --target-model=<t> \
        --target-path=<t> \
        --dest-source-path=<d> \
        --dest-target-path=<d> \
        [--max-size=<x>]
    tokenize.py train-shared --source-input-path=<s> \
        --target-input-path=<t> \
        --model-name=<t> \
        [--vocab-size=<s>] \
        [--input_sentence_size=<t>]
    tokenize.py tokenize-shared-bpe --task=<s> \
        --shared-model=<s> \
        --source-path=<s> \
        --target-path=<t> \
        --dest-source-path=<d> \
        --dest-target-path=<d> \
        [--max-size=<x>]

Options:
    --source-vocab-size=<s>             Source vocab size [default: 16_000].
    --source-input_sentence_size=<s>    Source input sentence size for batching [default: 1000].
    --target-vocab-size=<t>             Target vocab size [default: 16_000].
    --target-input_sentence_size=<t>    Target input sentence size for batching [default: 1000].
    --max-size=<x>                      Maximum number of output lines [default: -1].
    --vocab-size=<s>                    Shared vocab size [default: 16_000].
    --input_sentence_size=<t>           Shared vocab input sentence size [default: 1000].
"""

import os
import sys
import typing

import sentencepiece as spm
from docopt import docopt

from src.prepare_dataset import (
    preprocess_diff,
    preprocess_message,
    preprocess_src,
    preprocess_ast,
)
from src.utils import iterate_lines


class TextPreprocessor:
    def __init__(self):
        self.file_token = "<file>"
        self.chunk_token = "<chunk>"
        self.new_line_token = "<nl>"
        self.add_token = "<add>"
        self.del_token = "<del>"
        self.url_token = "<url>"
        self.number_token = "<num>"
        self.ref_token = "<ref>"
        self.sha_token = "<sha>"
        self.bos = "<s>"
        self.eos = "</s>"
        self.pad = "<pad>"
        self.unk = "<unk>"

    @property
    def symbols(self) -> typing.List[str]:
        return [
            self.file_token,
            self.chunk_token,
            self.new_line_token,
            self.add_token,
            self.del_token,
            self.url_token,
            self.number_token,
            self.ref_token,
            self.sha_token,
        ]


class Vocab:
    def __init__(
        self, symbols: typing.List[str], model_path: typing.Optional[str] = None
    ):
        self.symbols = symbols
        self.sentence_processor = spm.SentencePieceProcessor()
        if model_path:
            self.load(model_path)

    def train(
        self,
        input_path: str,
        model_name: str,
        vocab_size: int,
        input_sentence_size: int = 1000,
    ):
        symbols_str = ",".join(self.symbols)
        train_args = f"""--input={input_path} \
            --user_defined_symbols={symbols_str} \
            --model_prefix={model_name} \
            --pad_id=3 \
            --pad_piece=<pad> \
            --vocab_size={vocab_size} \
            --hard_vocab_limit={False} \
            --input_sentence_size={input_sentence_size} \
            --model_type=bpe"""

        spm.SentencePieceTrainer.Train(train_args)

    def train_shared(
        self,
        source_path: str,
        target_path: str,
        model_name: str,
        vocab_size: int,
        input_sentence_size: int = 1000,
    ):
        symbols_str = ",".join(self.symbols)
        train_args = f"""--input={source_path},{target_path} \
            --user_defined_symbols={symbols_str} \
            --model_prefix={model_name} \
            --pad_id=3 \
            --pad_piece=<pad> \
            --vocab_size={vocab_size} \
            --hard_vocab_limit={False} \
            --input_sentence_size={input_sentence_size} \
            --model_type=bpe"""

        spm.SentencePieceTrainer.Train(train_args)

    def load(self, model_path: str):
        self.sentence_processor.Load(model_path)

    def encode_as_pieces(self, sentence: str) -> typing.List[str]:
        return self.sentence_processor.encode_as_pieces(sentence)

    def decode_pieces(self, pieces: typing.List[str]) -> str:
        return self.sentence_processor.decode_pieces(pieces)

    def encode_as_ids(self, sentence: str) -> typing.List[int]:
        return self.sentence_processor.encode_as_ids(sentence)

    def decode_ids(self, ids: typing.List[int]) -> str:
        return self.sentence_processor.decode_ids(ids)

    def word2idx(self, word: str) -> int:
        return self.sentence_processor.piece_to_id(word)

    def idx2word(self, idx: int) -> str:
        return self.sentence_processor.id_to_piece(idx)

    def __len__(self) -> int:
        return len(self.sentence_processor)


class BPETokenizer:
    def __init__(
        self,
        source_vocab: Vocab,
        target_vocab: Vocab,
        source_preprocessor: typing.Optional[
            typing.Callable[[str], str]
        ] = None,
        target_preprocessor: typing.Optional[
            typing.Callable[[str], str]
        ] = None,
    ):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.source_preprocessor = source_preprocessor
        self.target_preprocessor = target_preprocessor

    def tokenize_source(self, sentence: str) -> typing.List[str]:
        if self.source_preprocessor:
            sentence = self.source_preprocessor(sentence)
        tokens = self.source_vocab.encode_as_pieces(sentence)
        return tokens

    def tokenize_target(self, sentence: str) -> typing.List[str]:
        if self.target_preprocessor:
            sentence = self.target_preprocessor(sentence)
        tokens = self.target_vocab.encode_as_pieces(sentence)
        return tokens

    def detokenize_source(self, tokens: typing.List[str]) -> str:
        sentence = self.source_vocab.decode_pieces(tokens)
        return sentence

    def detokenize_target(self, tokens: typing.List[str]) -> str:
        sentence = self.target_vocab.decode_pieces(tokens)
        return sentence

    def tokenize(
        self, source_sentence: str, target_sentence: str
    ) -> typing.Tuple[typing.List[str], typing.List[str]]:
        source_tokens = self.tokenize_source(source_sentence)
        target_tokens = self.tokenize_target(target_sentence)
        return source_tokens, target_tokens

    def detokenize(
        self, source_tokens: typing.List[str], target_tokens: typing.List[str]
    ) -> typing.Tuple[str, str]:
        source_sentence = self.detokenize_source(source_tokens)
        target_sentence = self.detokenize_target(target_tokens)
        return source_sentence, target_sentence

    def tokenize_source_as_ids(self, sentence: str) -> typing.List[int]:
        if self.target_preprocessor:
            sentence = self.source_preprocessor(sentence)
        ids = self.source_vocab.encode_as_ids(sentence)
        return ids

    def tokenize_target_as_ids(self, sentence: str) -> typing.List[int]:
        if self.target_preprocessor:
            sentence = self.target_preprocessor(sentence)
        ids = self.target_vocab.encode_as_ids(sentence)
        return ids

    def detokenize_source_ids(self, ids: typing.List[int]) -> str:
        sentence = self.source_vocab.decode_ids(ids)
        return sentence

    def detokenize_target_ids(self, ids: typing.List[int]) -> str:
        sentence = self.target_vocab.decode_ids(ids)
        return sentence

    def tokenize_as_ids(
        self, source_sentence: str, target_sentence: str
    ) -> typing.Tuple[typing.List[int], typing.List[int]]:
        source_ids = self.detokenize_source_ids(source_sentence)
        target_ids = self.detokenize_target_ids(target_sentence)
        return source_ids, target_ids

    def detokenize_ids(
        self, source_ids: typing.List[int], target_ids: typing.List[int]
    ) -> typing.Tuple[str, str]:
        source_sentence = self.detokenize_source_ids(source_ids)
        target_sentence = self.detokenize_target_ids(target_ids)
        return source_sentence, target_sentence


def __non_negative(value: int) -> int:
    if value < 0:
        return sys.maxsize
    return value


# pylint: disable=too-many-locals, too-many-arguments
def __tokenize_dataset(
    tokenizer: BPETokenizer,
    source_path: str,
    target_path: str,
    dest_source_path: str,
    dest_target_path: str,
    max_size: int = 1_000,
    max_len: int = -1,
) -> None:
    dest_dir = os.path.dirname(dest_source_path)
    os.makedirs(dest_dir, exist_ok=True)

    src_ptr = open(dest_source_path, mode="w")
    trg_ptr = open(dest_target_path, mode="w")
    corpus = zip(iterate_lines(source_path), iterate_lines(target_path))

    with src_ptr, trg_ptr:
        seen = 0
        for src, trg in corpus:
            if seen >= __non_negative(max_size):
                break
            src_tokens = tokenizer.tokenize_source(src)
            trg_tokens = tokenizer.tokenize_target(trg)

            if len(src_tokens) > __non_negative(max_len):
                continue
            if len(trg_tokens) > __non_negative(max_len):
                continue
            seen += 1
            src_ptr.write(" ".join(src_tokens) + "\n")
            trg_ptr.write(" ".join(trg_tokens) + "\n")


def __detokenize_dataset(
    tokenizer: BPETokenizer,
    source_path: str,
    target_path: str,
    dest_source_path: str,
    dest_target_path: str,
    max_size: int = -1,
) -> None:
    dest_dir = os.path.dirname(dest_source_path)
    os.makedirs(dest_dir, exist_ok=True)

    src_ptr = open(dest_source_path, mode="w")
    trg_ptr = open(dest_target_path, mode="w")
    corpus = zip(iterate_lines(source_path), iterate_lines(target_path))

    with src_ptr, trg_ptr:
        seen = 0
        for src, trg in corpus:
            if seen >= __non_negative(max_size):
                break

            src = src.split("\n")[0]
            trg = trg.split("\n")[0]
            src_sent = tokenizer.detokenize_source(src.split(" "))
            trg_sent = tokenizer.detokenize_target(trg.split(" "))

            seen += 1
            src_ptr.write("".join(src_sent) + "\n")
            trg_ptr.write("".join(trg_sent) + "\n")


def tokenize_dataset(
    symbols: typing.List[str],
    source_model_path: str,
    target_model_path: str,
    source_preprocessor: typing.Optional[typing.Callable[[str], str]],
    target_preprocessor: typing.Optional[typing.Callable[[str], str]],
    source_path: str,
    target_path: str,
    dest_source_path: str,
    dest_target_path: str,
    max_size: int = 1_000,
    max_len: int = -1,
) -> None:
    source_vocab = Vocab(symbols)
    target_vocab = Vocab(symbols)
    source_vocab.load(source_model_path)
    target_vocab.load(target_model_path)

    tokenizer = BPETokenizer(
        source_vocab, target_vocab, source_preprocessor, target_preprocessor,
    )

    __tokenize_dataset(
        tokenizer,
        source_path,
        target_path,
        dest_source_path,
        dest_target_path,
        max_size,
        max_len,
    )


def tokenize_shared_dataset(
    symbols: typing.List[str],
    shared_model_path: str,
    source_preprocessor: typing.Optional[typing.Callable[[str], str]],
    target_preprocessor: typing.Optional[typing.Callable[[str], str]],
    source_path: str,
    target_path: str,
    dest_source_path: str,
    dest_target_path: str,
    max_size: int = 1_000,
    max_len: int = -1,
) -> None:
    shared_vocab: Vocab = Vocab(symbols, model_path=shared_model_path)

    tokenizer = BPETokenizer(
        source_vocab=shared_vocab,
        target_vocab=shared_vocab,
        source_preprocessor=source_preprocessor,
        target_preprocessor=target_preprocessor,
    )

    __tokenize_dataset(
        tokenizer,
        source_path,
        target_path,
        dest_source_path,
        dest_target_path,
        max_size,
        max_len,
    )


def train_vocabs(
    symbols: typing.List[str],
    source_input_path: str,
    source_model_name: str,
    source_vocab_size: int,
    source_input_sentence_size: int,
    target_input_path: str,
    target_model_name: str,
    target_vocab_size: int,
    target_input_sentence_size: int,
):
    source_vocab = Vocab(symbols)
    source_vocab.train(
        input_path=source_input_path,
        model_name=source_model_name,
        vocab_size=source_vocab_size,
        input_sentence_size=source_input_sentence_size,
    )

    target_vocab = Vocab(symbols)
    target_vocab.train(
        input_path=target_input_path,
        model_name=target_model_name,
        vocab_size=target_vocab_size,
        input_sentence_size=target_input_sentence_size,
    )


def train_shared_vocab(
    symbols: typing.List[str],
    source_input_path: str,
    target_input_path: str,
    shared_model_name: str,
    shared_vocab_size: int,
    input_sentence_size: int,
):
    shared_vocab: Vocab = Vocab(symbols)
    shared_vocab.train_shared(
        source_path=source_input_path,
        target_path=target_input_path,
        model_name=shared_model_name,
        vocab_size=shared_vocab_size,
        input_sentence_size=input_sentence_size,
    )


def base_tokenizer(source_vocab: Vocab, target_vocab: Vocab) -> BPETokenizer:
    tokenizer = BPETokenizer(
        source_vocab,
        target_vocab,
        source_preprocessor=preprocess_diff,
        target_preprocessor=preprocess_message,
    )

    return tokenizer


def main():
    """
    Usage examples:

    python -m src.tokenize train \
        --source-input-path=src-path \
        --source-model-name=src-name \
        --target-input-path=trg-path \
        --target-model-name=trg-name

    python -m src.tokenize tokenize-bpe \
        --source-model=/workspace/diffs.model \
        --source-path=/workspace/tmp/train.diff \
        --target-model=/workspace/messages.model \
        --target-path=/workspace/tmp/train.msg \
        --dest-source-path=/workspace/tmp/dataset/train.diff \
        --dest-target-path=/workspace/tmp/dataset/train.msg

    python -m src.tokenize train-shared \
        --source-input-path=/workspace/tmp/train.diff \
        --target-input-path=/workspace/tmp/train.msg \
        --model-name=sentencepiece \
        --vocab-size=32000

    python -m src.tokenize tokenize-shared-bpe \
        --shared-model=/workspace/sentencepiece.model \
        --source-path=/workspace/tmp/train.diff \
        --target-path=/workspace/tmp/train.msg \
        --dest-source-path=/workspace/tmp/dataset_200k.bpe/train.diff \
        --dest-target-path=/workspace/tmp/dataset_200k.bpe/train.msg \
        --max-size=200000
    """
    symbols = TextPreprocessor().symbols
    arguments = docopt(__doc__, version="Tokenization 1.0")

    diff2msg_task = "diff2msg"
    code2ast_task = "code2ast"

    if arguments["train"]:
        source_input_path = arguments["--source-input-path"]
        source_model_name = arguments["--source-model-name"]
        source_vocab_size = arguments["--source-vocab-size"]
        source_input_sentence_size = arguments["--source-input_sentence_size"]

        target_input_path = arguments["--target-input-path"]
        target_model_name = arguments["--target-model-name"]
        target_vocab_size = arguments["--target-vocab-size"]
        target_input_sentence_size = arguments["--target-input_sentence_size"]

        train_vocabs(
            symbols,
            source_input_path=source_input_path,
            source_model_name=source_model_name,
            source_vocab_size=source_vocab_size,
            source_input_sentence_size=source_input_sentence_size,
            target_input_path=target_input_path,
            target_model_name=target_model_name,
            target_vocab_size=target_vocab_size,
            target_input_sentence_size=target_input_sentence_size,
        )

    elif arguments["train-shared"]:
        source_path = arguments["--source-input-path"]
        target_path = arguments["--target-input-path"]
        model_name = arguments["--model-name"]
        vocab_size = arguments["--vocab-size"]
        input_sentence_size = arguments["--input_sentence_size"]

        train_shared_vocab(
            symbols,
            source_input_path=source_path,
            target_input_path=target_path,
            shared_model_name=model_name,
            shared_vocab_size=vocab_size,
            input_sentence_size=input_sentence_size,
        )

    elif arguments["tokenize-bpe"]:
        source_model_path = arguments["--source-model"]
        target_model_path = arguments["--target-model"]
        source_path = arguments["--source-path"]
        target_path = arguments["--target-path"]
        dest_source_path = arguments["--dest-source-path"]
        dest_target_path = arguments["--dest-target-path"]
        max_size = int(arguments["--max-size"])

        os.makedirs(os.path.dirname(dest_source_path), exist_ok=True)
        os.makedirs(os.path.dirname(dest_target_path), exist_ok=True)

        task = arguments["--task"]
        if task == diff2msg_task:
            source_preprocessor = preprocess_diff
            target_preprocessor = preprocess_message
        elif task == code2ast_task:
            source_preprocessor = preprocess_src
            target_preprocessor = preprocess_ast
        else:
            source_preprocessor = None
            target_preprocessor = None

        tokenize_dataset(
            symbols,
            source_model_path=source_model_path,
            target_model_path=target_model_path,
            source_preprocessor=source_preprocessor,
            target_preprocessor=target_preprocessor,
            source_path=source_path,
            target_path=target_path,
            dest_source_path=dest_source_path,
            dest_target_path=dest_target_path,
            max_size=max_size,
            max_len=512,
        )

    elif arguments["tokenize-shared-bpe"]:
        shared_model_path = arguments["--shared-model"]
        source_path = arguments["--source-path"]
        target_path = arguments["--target-path"]
        dest_source_path = arguments["--dest-source-path"]
        dest_target_path = arguments["--dest-target-path"]
        max_size = int(arguments["--max-size"])

        task = arguments["--task"]
        if task == diff2msg_task:
            source_preprocessor = preprocess_diff
            target_preprocessor = preprocess_message
        elif task == code2ast_task:
            source_preprocessor = preprocess_src
            target_preprocessor = preprocess_ast
        else:
            source_preprocessor = None
            target_preprocessor = None

        tokenize_shared_dataset(
            symbols,
            shared_model_path=shared_model_path,
            source_preprocessor=source_preprocessor,
            target_preprocessor=target_preprocessor,
            source_path=source_path,
            target_path=target_path,
            dest_source_path=dest_source_path,
            dest_target_path=dest_target_path,
            max_size=max_size,
            max_len=512,
        )


if __name__ == "__main__":
    main()
