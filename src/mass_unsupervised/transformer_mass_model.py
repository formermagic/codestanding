import typing
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn.functional as F

from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.tasks import FairseqTask
from fairseq.utils import (
    get_available_activation_fns,
    parse_embedding,
    load_embedding,
)

from .layer_initialization import Embedding, Linear
from .masked_dictionary import MaskedDictionary
from .transformer_decoder import TransformerDecoder
from .transformer_encoder import TransformerEncoder
from .unsupervised_mass_task import UnsupervisedMASSTask

try:
    from typing import OrderedDict as OrderedDictType
except ImportError:
    from typing import MutableMapping as OrderedDictType

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512

Extras = Dict[str, Union[torch.Tensor, List[Optional[torch.Tensor]]]]
DecoderOutput = Tuple[torch.Tensor, Extras]


@register_model("transformer_mass")
class TransformerMASSModel(FairseqEncoderDecoderModel):
    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        langs: List[str],
    ) -> None:
        super().__init__(encoder, decoder)
        self.lang2idx = {}
        self.idx2lang = {}
        for idx, lang in enumerate(langs):
            self.lang2idx[lang] = idx
            self.idx2lang[idx] = lang

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        # fmt: off
        parser.add_argument("--activation-fn", choices=get_available_activation_fns(),
                            help="activation function to use")
        parser.add_argument("--dropout", type=float, metavar="D", 
                            help="dropout probability")
        parser.add_argument("--attention-dropout", type=float, metavar="D",
                            help="dropout probability for attention weights")
        parser.add_argument("--activation-dropout", type=float, metavar="D",
                            help="dropout probability after activation in FFN.")

        parser.add_argument("--encoder-embed-dim", type=int, metavar="N",
                            help="encoder embedding dimension")
        parser.add_argument("--encoder-ffn-embed-dim", type=int, metavar="N",
                            help="encoder embedding dimension for FFN")
        parser.add_argument("--encoder-layers", type=int, metavar="N", 
                            help="num encoder layers")
        parser.add_argument("--encoder-attention-heads", type=int, metavar="N",
                            help="num encoder attention heads")

        parser.add_argument("--decoder-embed-dim", type=int, metavar="N",
                            help="decoder embedding dimension")
        parser.add_argument("--decoder-ffn-embed-dim", type=int, metavar="N",
                            help="decoder embedding dimension for FFN")
        parser.add_argument("--decoder-layers", type=int, metavar="N", 
                            help="num decoder layers")
        parser.add_argument("--decoder-attention-heads", type=int, metavar="N",
                            help="num decoder attention heads")

        parser.add_argument("--share-encoder-embeddings", action="store_true",
                            help="share encoder embeddings across languages")
        parser.add_argument("--share-decoder-embeddings", action="store_true",
                            help="share decoder embeddings across languages")
        parser.add_argument("--share-encoders", action="store_true",
                            help="share encoders across languages")
        parser.add_argument("--share-decoders", action="store_true",
                            help="share decoders across languages")
        parser.add_argument("--share-all-embeddings", action="store_true",
                            help="share encoder, decoder and output embeddings"
                                " (requires shared dictionary and embed dim)")

        parser.add_argument("--load-from-pretrained-model", type=str, default=None,
                            help="Load from pretrained model")

        parser.add_argument("--decoder-embed-path", type=str, metavar="STR",
                            help="path to pre-trained decoder embedding")
        # fmt: on

    @classmethod
    def build_embedding(
        cls,
        dictionary: MaskedDictionary,
        embedding_dim: int,
        pretrained_embedding_path: Optional[str] = None,
    ) -> Embedding:
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        embedding = Embedding(num_embeddings, embedding_dim, padding_idx)
        if pretrained_embedding_path:
            embedding_dict = parse_embedding(pretrained_embedding_path)
            load_embedding(embedding_dict, dictionary, embedding)
        return embedding

    @classmethod
    def build_shared_embeddings(
        cls,
        dicts: Dict[str, MaskedDictionary],
        langs: List[str],
        embedding_dim: int,
        pretrained_embedding_path: Optional[str] = None,
    ) -> Embedding:
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                "--share-*-embeddings requires a joined dictionary: "
                "--share-encoder-embeddings requires a joined source "
                "dictionary, --share-decoder-embeddings requires a joined "
                "target dictionary, and --share-all-embeddings requires a "
                "joint source + target dictionary."
            )
        return cls.build_embedding(
            shared_dict, embedding_dim, pretrained_embedding_path
        )

    @classmethod
    def build_encoder(
        cls,
        args: Namespace,
        langs: int,
        dictionary: MaskedDictionary,
        shared_embedding_tokens: Optional[Embedding],
    ) -> TransformerEncoder:
        if shared_embedding_tokens is not None:
            embedding_tokens = shared_embedding_tokens
        else:
            embedding_tokens = cls.build_embedding(
                dictionary, args.encoder_embed_dim, args.encoder_embed_path
            )

        embedding_languages = Embedding(langs, args.encoder_embed_dim)

        return TransformerEncoder(
            dictionary,
            args.dropout,
            args.max_source_positions,
            args.encoder_layers,
            embedding_languages,
            embedding_tokens,
            args.encoder_ffn_embed_dim,
            args.encoder_attention_heads,
            args.attention_dropout,
            args.activation_dropout,
            args.activation_fn,
        )

    @classmethod
    def build_decoder(
        cls,
        args: Namespace,
        langs: int,
        dictionary: MaskedDictionary,
        shared_embedding_tokens: Optional[Embedding],
    ) -> TransformerDecoder:
        if shared_embedding_tokens is not None:
            embedding_tokens = shared_embedding_tokens
        else:
            embedding_tokens = cls.build_embedding(
                dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )

        embedding_languages = Embedding(langs, args.decoder_embed_dim)

        return TransformerDecoder(
            dictionary,
            args.dropout,
            args.max_target_positions,
            args.decoder_layers,
            embedding_languages,
            embedding_tokens,
            args.encoder_embed_dim,
            args.decoder_ffn_embed_dim,
            args.decoder_attention_heads,
            args.attention_dropout,
            args.activation_dropout,
            args.activation_fn,
            args.share_decoder_input_output_embed,
        )

    @classmethod
    def build_model(
        cls, args: Namespace, task: FairseqTask
    ) -> "TransformerMASSModel":
        assert isinstance(task, UnsupervisedMASSTask)
        base_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_langs = args.source_langs
        tgt_langs = args.target_langs

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        shared_encoder_embedding_tokens: Optional[Embedding] = None
        shared_decoder_embedding_tokens: Optional[Embedding] = None

        if args.share_all_embeddings:
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible "
                    "with --decoder-embed-path"
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires "
                    "--encoder-embed-dim to match --decoder-embed-dim"
                )

            shared_encoder_embedding_tokens = cls.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embedding_dim=args.encoder_embed_dim,
                pretrained_embedding_path=args.encoder_embed_path,
            )
            shared_decoder_embedding_tokens = shared_encoder_embedding_tokens

            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embedding_tokens = cls.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=src_langs,
                    embedding_dim=args.encoder_embed_dim,
                    pretrained_embedding_path=args.encoder_embed_path,
                )

            if args.share_decoder_embeddings:
                shared_decoder_embedding_tokens = cls.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embedding_dim=args.decoder_embed_dim,
                    pretrained_embedding_path=args.decoder_embed_path,
                )

        langs = len(task.langs)
        encoder = cls.build_encoder(
            args, langs, task.source_dictionary, shared_encoder_embedding_tokens
        )
        decoder = cls.build_decoder(
            args, langs, task.target_dictionary, shared_decoder_embedding_tokens
        )

        model = TransformerMASSModel(encoder, decoder, args.langs)

        if args.load_from_pretrained_model is not None:
            state_dict = torch.load(
                args.load_from_pretrained_model, map_location="cpu"
            )

            model.load_state_dict(state_dict)
            args.load_from_pretrained_model = None

        return model

    def max_positions(self) -> OrderedDictType[str, Tuple[int, int]]:
        """Use a workaround to correctly resolve max positions"""
        positions = self.encoder.max_positions(), self.decoder.max_positions()
        return OrderedDict([("model", positions)])

    def max_decoder_positions(self) -> int:
        return self.decoder.max_positions()

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: torch.Tensor,
        lang_pair: str,
        **kwargs: Any,
    ) -> DecoderOutput:
        src_lang, tgt_lang = lang_pair.split("-")
        src_idx = self.lang2idx[src_lang]
        tgt_idx = self.lang2idx[tgt_lang]

        # shape: [Batch, Time]
        src_langs = torch.LongTensor([src_idx]).expand_as(src_tokens)
        tgt_langs = torch.LongTensor([tgt_idx]).expand_as(prev_output_tokens)

        encoder_output = self.encoder(
            src_tokens, src_lengths, languages=src_langs, **kwargs
        )
        decoder_output = self.decoder(
            prev_output_tokens, encoder_output, languages=tgt_langs, **kwargs
        )

        return decoder_output

    def forward_decoder(
        self, prev_output_tokens: torch.Tensor, **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.decoder(prev_output_tokens, **kwargs)

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: DecoderOutput,
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.get_normalized_probs_scriptable(
            net_output, log_probs, sample
        )

    def get_normalized_probs_scriptable(
        self,
        net_output: DecoderOutput,
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(
                net_output, log_probs, sample
            )
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)

        raise NotImplementedError

    def get_targets(
        self, sample: Dict[str, torch.Tensor], net_output: DecoderOutput
    ) -> torch.Tensor:
        """Get targets from either the sample or the net's output."""
        tgt = sample["target"]
        return tgt


@register_model_architecture("transformer_mass", "transformer_mass_cased")
def base_architecture(args: Namespace) -> None:
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)

    args.share_encoder_embeddings = getattr(
        args, "share_encoder_embeddings", False
    )
    args.share_decoder_embeddings = getattr(
        args, "share_decoder_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.share_encoders = getattr(args, "share_encoders", False)
    args.share_decoders = getattr(args, "share_decoders", False)


@register_model_architecture("transformer_mass", "transformer_mass_base")
def transformer_base(args: Namespace) -> None:
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)

    base_architecture(args)


@register_model_architecture("transformer_mass", "transformer_mass_middle")
def transformer_middle(args: Namespace) -> None:
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)

    transformer_base(args)


@register_model_architecture("transformer_mass", "transformer_mass_big")
def transformer_big(args: Namespace) -> None:
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 12)

    transformer_middle(args)
