from fairseq.data.encoders import register_bpe


@register_bpe("bert-wordpiece")
class BertWordPieceBPE(object):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--bpe-cased', action='store_true',
                            help='set for cased BPE',
                            default=False)
        parser.add_argument('--bpe-vocab-file', type=str,
                            help='bpe vocab file.')
        # fmt: on

    def __init__(self, args):
        try:
            from transformers import BertTokenizer
        except ImportError:
            raise ImportError(
                "Please install 2.8.0 version of transformers"
                "with: pip install transformers==2.8.0"
            )

        if "bpe_vocab_file" in args:
            self.bert_tokenizer = BertTokenizer(
                args.bpe_vocab_file, do_lower_case=not args.bpe_cased
            )
        else:
            vocab_file_name = (
                "bert-base-cased" if args.bpe_cased else "bert-base-uncased"
            )
            self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file_name)

        self.clean_up_tokenization = BertTokenizer.clean_up_tokenization

    def encode(self, x: str) -> str:
        return " ".join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        return self.clean_up_tokenization(
            self.bert_tokenizer.convert_tokens_to_string(x.split(" "))
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith("##")
