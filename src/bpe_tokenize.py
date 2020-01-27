"""Code tokenization pipelines.

Usage:
    bpe_tokenize.py train \
        --file-one=<file_one> \
        --file-two=<file_two> \
        --vocab-size=<voc_size> \
        --min-frequency=<freq> \
        --output-suffix=<output_suffix>
    bpe_tokenize.py tokenize \
        --vocab-file=<voc_file> \
        --merges-file=<merges_file> \
        --input-file-one=<inp_one> \
        --input-file-two=<inp_two> \
        --output-file-one=<out_one> \
        --output-file-two=<out_two>
    bpe_tokenize.py detokenize \
        --vocab-file=<voc_file> \
        --merges-file=<merges_file> \
        --input-file-one=<inp_one> \
        --input-file-two=<inp_two> \
        --output-file-one=<out_one> \
        --output-file-two=<out_two>

Options:
    --file-one=<file_one>               A path to the first file to train a BPE model on.
    --file-two=<file_two>               A path to the second file to train a BPE model on.
    --vocab-size=<voc_size>             Tokenizer max vocabulary size.
    --min-frequency=<freq>              Tokenizer min frequency of a word.
    --output-suffix=<output_suffix>     An output suffix for the trained model.
    --vocab-file=<voc_file>             A path to the trained vocab file.
    --merges-file=<merges_file>         A path to the trained merges file.
    --input-file-one=<inp_one>          A path to the first input file to process.
    --input-file-two=<inp_two>          A path to the second input file to process.
    --output-file-one=<out_one>         A path to the first output file to write to.
    --output-file-two=<out_two>         A path to the second output file to write to.
"""
import os
from io import FileIO
from typing import List, Optional, Tuple, Union

from docopt import docopt
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.implementations import BaseTokenizer
from tokenizers.normalizers import Lowercase, Sequence

from .utils import iterate_lines


class CodeBPETokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: Optional[str] = "<unk>",
        suffix: Optional[str] = "</w>",
        dropout: Optional[float] = None,
    ):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(
                models.BPE.from_files(
                    vocab_file,
                    merges_file,
                    dropout=dropout,
                    unk_token=unk_token,
                    end_of_word_suffix=suffix,
                )
            )
        else:
            tokenizer = Tokenizer(models.BPE.empty())

        tokenizer.normalizer = Sequence.new([Lowercase.new()])
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit.new()
        tokenizer.decoder = decoders.BPEDecoder.new(suffix=suffix)

        tokenizer.add_special_tokens(["<nl>"])
        tokenizer.add_tokens(["<nl>"])

        parameters = {
            "model": "CodeBPE",
            "unk_token": unk_token,
            "suffix": suffix,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30_000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        suffix: Optional[str] = "</w>",
        show_progress: bool = True,
    ):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer.new(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            end_of_word_suffix=suffix,
            show_progress=show_progress,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)


def train_tokenizer(
    files: List[str], vocab_size: int, min_frequency: int, output_suffix: str
) -> BaseTokenizer:
    output_dir = os.path.dirname(output_suffix)
    output_name = os.path.basename(output_suffix)
    output_name = os.path.splitext(output_name)[0]

    tokenizer = CodeBPETokenizer()
    tokenizer.train(files, vocab_size, min_frequency)
    tokenizer.save(output_dir, output_name)

    return tokenizer


class TokenBufferWriter:
    def __init__(self, buffer_threshold: int = 15_000) -> None:
        self.buffer: List[str] = []
        self.buffer_threshold = buffer_threshold

    def write_tokens(
        self, tokens: List[str], output_file: FileIO, force_flush: bool = False,
    ) -> None:
        self.buffer += tokens
        if len(self.buffer) >= self.buffer_threshold or force_flush:
            output_file.writelines(self.buffer)
            self.buffer.clear()


class PairTokenBufferWriter:
    def __init__(self, buffer_threshold: int = 15_000) -> None:
        self.writer_one = TokenBufferWriter(buffer_threshold)
        self.writer_two = TokenBufferWriter(buffer_threshold)

    def write_tokens(
        self,
        tokens: Tuple[List[str], List[str]],
        output_files: Tuple[FileIO, FileIO],
        force_flush: bool = False,
    ) -> None:
        tokens_one, tokens_two = tokens
        file_one, file_two = output_files
        self.writer_one.write_tokens(tokens_one, file_one, force_flush)
        self.writer_two.write_tokens(tokens_two, file_two, force_flush)


class PairTokenizerWrapper:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        buffer_writer: PairTokenBufferWriter,
        batch_size: int = 1024,
    ) -> None:
        self.tokenizer = tokenizer
        self.buffer_writer = buffer_writer
        self.batch_size = batch_size
        self.batch: List[Tuple[str, str]] = []

    def __batch_encode(self, batch: List[str]) -> List[str]:
        encodings = self.tokenizer.encode_batch(batch)
        tokens_sequence = [
            " ".join(encoding.tokens) + "\n" for encoding in encodings
        ]
        return tokens_sequence

    def batch_encode(self) -> Tuple[List[str], List[str]]:
        batch_one, batch_two = zip(*self.batch)
        batch_tokens = zip(
            self.__batch_encode(list(batch_one)),
            self.__batch_encode(list(batch_two)),
        )

        max_len = 512
        sequences_one: List[str] = []
        sequences_two: List[str] = []

        for tokens_one, tokens_two in batch_tokens:
            lsrger_lengths = [
                len(tokens_one.split(" ")) > max_len,
                len(tokens_two.split(" ")) > max_len,
            ]
            if any(lsrger_lengths):
                continue
            sequences_one.append(tokens_one)
            sequences_two.append(tokens_two)

        self.batch.clear()

        return sequences_one, sequences_two

    def pair_batch_encode(
        self, input_files: Tuple[str, str], output_files: Tuple[str, str]
    ) -> None:
        for output_path in output_files:
            if os.path.exists(output_path):
                os.remove(output_path)

        input_lines = zip(
            iterate_lines(input_files[0]), iterate_lines(input_files[1])
        )

        output_file_one = open(output_files[0], mode="w")
        output_file_two = open(output_files[1], mode="w")

        def process_batch(force_flush: bool = False) -> None:
            seq_one, seq_two = self.batch_encode()
            self.buffer_writer.write_tokens(
                tokens=(seq_one, seq_two),
                output_files=(output_file_one, output_file_two),
                force_flush=force_flush,
            )

        with output_file_one, output_file_two:
            for line_one, line_two in input_lines:
                if len(self.batch) >= self.batch_size:
                    process_batch()
                self.batch.append((line_one, line_two))

            if self.batch:
                process_batch(force_flush=True)

    def __batch_decode(self, batch: List[str]) -> List[str]:
        batch_ids = [
            [int(idx) for idx in idx_line.split(" ")] for idx_line in batch
        ]
        decoded_lines = [
            line + "\n"
            for line in self.tokenizer.decode_batch(
                batch_ids, skip_special_tokens=False
            )
        ]
        return decoded_lines

    def batch_decode(self) -> Tuple[List[str], List[str]]:
        batch_one, batch_two = zip(*self.batch)
        decoded_lines_one: List[str] = self.__batch_decode(batch_one)
        decoded_lines_two: List[str] = self.__batch_decode(batch_two)
        self.batch.clear()

        return decoded_lines_one, decoded_lines_two

    def pair_batch_decode(
        self, input_files: Tuple[str, str], output_files: Tuple[str, str]
    ) -> None:
        for output_path in output_files:
            if os.path.exists(output_path):
                os.remove(output_path)

        input_lines = zip(
            iterate_lines(input_files[0]), iterate_lines(input_files[1])
        )

        output_file_one = open(output_files[0], mode="w")
        output_file_two = open(output_files[1], mode="w")

        def process_batch(force_flush: bool = False) -> None:
            seq_one, seq_two = self.batch_decode()
            self.buffer_writer.write_tokens(
                tokens=(seq_one, seq_two),
                output_files=(output_file_one, output_file_two),
                force_flush=force_flush,
            )

        with output_file_one, output_file_two:
            for line_one, line_two in input_lines:
                if len(self.batch) >= self.batch_size:
                    process_batch()

                line_one_ids = [
                    str(self.tokenizer.token_to_id(token))
                    for token in line_one.strip().split(" ")
                ]
                line_two_ids = [
                    str(self.tokenizer.token_to_id(token))
                    for token in line_two.strip().split(" ")
                ]

                line_one_ids = " ".join(line_one_ids)
                line_two_ids = " ".join(line_two_ids)

                self.batch.append((line_one_ids, line_two_ids))

            if self.batch:
                process_batch(force_flush=True)


def main() -> None:
    """
    python -m src.bpe_tokenize train \
        --file-one=/workspace/tmp/code2ast/samples/all.src \
        --file-two=/workspace/tmp/code2ast/samples/all.ast \
        --vocab-size=64000 \
        --min-frequency=2 \
        --output-suffix=/workspace/tmp/code2ast/samples/voc_64k_2

    python -m src.bpe_tokenize tokenize \
        --vocab-file=/workspace/tmp/code2ast/samples/voc_64k-vocab.json \
        --merges-file=/workspace/tmp/code2ast/samples/voc_64k-merges.txt \
        --input-file-one=/workspace/tmp/code2ast/samples/all.src \
        --input-file-two=/workspace/tmp/code2ast/samples/all.ast \
        --output-file-one=/workspace/tmp/code2ast/samples/all.src.tok \
        --output-file-two=/workspace/tmp/code2ast/samples/all.ast.tok

    python -m src.bpe_tokenize detokenize \
        --vocab-file=/workspace/tmp/code2ast/samples/voc_64k-vocab.json \
        --merges-file=/workspace/tmp/code2ast/samples/voc_64k-merges.txt \
        --input-file-one=/workspace/tmp/code2ast/samples/all.src.tok \
        --input-file-two=/workspace/tmp/code2ast/samples/all.ast.tok \
        --output-file-one=/workspace/tmp/code2ast/samples/all.src.detok \
        --output-file-two=/workspace/tmp/code2ast/samples/all.ast.detok
    """
    arguments = docopt(__doc__, version="Code Tokenization 1.0")

    if arguments["train"]:
        file_one = str(arguments["--file-one"])
        file_two = str(arguments["--file-two"])
        vocab_size = int(arguments["--vocab-size"])
        min_frequency = int(arguments["--min-frequency"])
        output_suffix = str(arguments["--output-suffix"])

        files = (file_one, file_two)
        train_tokenizer(files, vocab_size, min_frequency, output_suffix)
    else:
        vocab_file = str(arguments["--vocab-file"])
        merges_file = str(arguments["--merges-file"])
        input_file_one = str(arguments["--input-file-one"])
        input_file_two = str(arguments["--input-file-two"])
        output_file_one = str(arguments["--output-file-one"])
        output_file_two = str(arguments["--output-file-two"])

        input_files = (input_file_one, input_file_two)
        output_files = (output_file_one, output_file_two)

        tokenizer = CodeBPETokenizer(vocab_file, merges_file)
        buffer_writer = PairTokenBufferWriter()
        pair_tokenizer = PairTokenizerWrapper(tokenizer, buffer_writer)

        if arguments["tokenize"]:
            pair_tokenizer.pair_batch_encode(
                input_files, output_files,
            )
        elif arguments["detokenize"]:
            pair_tokenizer.pair_batch_decode(
                input_files, output_files,
            )


if __name__ == "__main__":
    main()
