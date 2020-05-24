from argparse import ArgumentParser

from .indexed_dataset import IndexedDataset, IndexDatasetPreprocessor
from .tokenization_codebert import CodeBertTokenizerFast


def main() -> None:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="A path to pretrained tokenizer saved files.")
    parser.add_argument("--file_path", type=str, default=None,
                        help="A path to the dataset file to index and save.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="A path to the output dir to save indexed file to.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="The maximum length of lines to be processed.")
    # fmt: on

    args = parser.parse_args()
    tokenizer_path = args.tokenizer_path
    filepath = args.file_path
    output_file = args.output_path
    max_length = args.max_length

    tokenizer = CodeBertTokenizerFast.from_pretrained(tokenizer_path)
    preprocessor = IndexDatasetPreprocessor(filepath, tokenizer, max_length)
    preprocessor.preprocess(output_file)


if __name__ == "__main__":
    main()
