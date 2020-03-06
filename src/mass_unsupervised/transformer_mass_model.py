import typing
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional, Union

import torch
import torch.nn.functional as F
from fairseq.models import (
    FairseqEncoderDecoderModel,
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.tasks import FairseqTask
from fairseq.utils import get_available_activation_fns

from .layer_initialization import Embedding, Linear
from .masked_dictionary import MaskedDictionary
from .transformer_decoder import TransformerDecoder
from .transformer_encoder import TransformerEncoder
from .unsupervised_mass_task import UnsupervisedMASSTask

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512

DecoderOutput = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


@register_model("transformer_mass")
class TransformerMASSModel(FairseqMultiModel):
    def __init__(
        self,
        encoders: typing.OrderedDict[str, TransformerEncoder],
        decoders: typing.OrderedDict[str, TransformerDecoder],
    ) -> None:
        super().__init__(encoders, decoders)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--activation-fn",
            choices=get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )

        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )

        parser.add_argument(
            "--share-encoder-embeddings",
            action="store_true",
            help="share encoder embeddings across languages",
        )
        parser.add_argument(
            "--share-decoder-embeddings",
            action="store_true",
            help="share decoder embeddings across languages",
        )
        parser.add_argument(
            "--share-encoders",
            action="store_true",
            help="share encoders across languages",
        )
        parser.add_argument(
            "--share-decoders",
            action="store_true",
            help="share decoders across languages",
        )
        parser.add_argument(
            "--share-all-embeddings",
            action="store_true",
            help="share encoder, decoder and output embeddings"
            " (requires shared dictionary and embed dim)",
        )

        parser.add_argument(
            "--load-from-pretrained-model",
            type=str,
            default=None,
            help="Load from pretrained model",
        )

        parser.add_argument(
            "--decoder-embed-path",
            type=str,
            metavar="STR",
            help="path to pre-trained decoder embedding",
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

        def build_embedding(
            dictionary: MaskedDictionary,
            embedding_dim: int,
            embedding_path: Optional[str] = None,
        ) -> Embedding:
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embedding = Embedding(num_embeddings, embedding_dim, padding_idx)
            return embedding

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

            shared_encoder_embedding_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embedding_tokens = shared_encoder_embedding_tokens

            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embedding_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=src_langs,
                    embed_dim=args.encoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.encoder_embed_path,
                )

            if args.share_decoder_embeddings:
                shared_decoder_embedding_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.decoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.decoder_embed_path,
                )

        lang_encoders: Dict[str, TransformerEncoder] = {}
        lang_decoders: Dict[str, TransformerDecoder] = {}

        def build_encoder(
            lang: str, encoders: Dict[str, TransformerEncoder]
        ) -> TransformerEncoder:
            if lang not in encoders:
                if shared_encoder_embedding_tokens is not None:
                    encoder_embedding_tokens = shared_encoder_embedding_tokens
                else:
                    encoder_embedding_tokens = build_embedding(
                        task.dicts[lang], args.encoder_embed_dim
                    )

                encoders[lang] = TransformerEncoder(
                    task.dicts[lang],
                    args.dropout,
                    args.max_source_positions,
                    args.encoder_layers,
                    encoder_embedding_tokens,
                    args.encoder_ffn_embed_dim,
                    args.encoder_attention_heads,
                    args.attention_dropout,
                    args.activation_dropout,
                    args.activation_fn,
                )

            return encoders[lang]

        def build_decoder(
            lang: str, decoders: Dict[str, TransformerEncoder]
        ) -> TransformerDecoder:
            if lang not in decoders:
                if shared_decoder_embedding_tokens is not None:
                    decoder_embedding_tokens = shared_decoder_embedding_tokens
                else:
                    decoder_embedding_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim
                    )

                decoders[lang] = TransformerDecoder(
                    task.dicts[lang],
                    args.dropout,
                    args.max_target_positions,
                    args.decoder_layers,
                    decoder_embedding_tokens,
                    args.encoder_embed_dim,
                    args.decoder_ffn_embed_dim,
                    args.decoder_attention_heads,
                    args.attention_dropout,
                    args.activation_dropout,
                    args.activation_fn,
                    args.share_decoder_input_output_embed,
                )

            return decoders[lang]

        shared_encoder: Optional[TransformerEncoder] = None
        shared_decoder: Optional[TransformerDecoder] = None

        if args.share_encoders:
            shared_encoder = build_encoder(
                lang=src_langs[0], encoders=lang_encoders
            )
        if args.share_decoders:
            shared_decoder = build_decoder(
                lang=tgt_langs[0], decoders=lang_decoders
            )

        encoders: typing.OrderedDict[str, TransformerEncoder] = OrderedDict()
        decoders: typing.OrderedDict[str, TransformerDecoder] = OrderedDict()

        for lang_pair in task.lang_pairs:
            src_lang, tgt_lang = lang_pair.split("-")[:2]
            if shared_encoder is not None:
                encoders[lang_pair] = shared_encoder
            else:
                encoders[lang_pair] = build_encoder(
                    lang=src_lang, encoders=lang_encoders
                )

            if shared_decoder is not None:
                decoders[lang_pair] = shared_decoder
            else:
                decoders[lang_pair] = build_decoder(
                    lang=tgt_lang, decoders=lang_decoders
                )

        model = TransformerMASSModel(encoders, decoders)

        if args.load_from_pretrained_model is not None:
            state_dict = torch.load(
                args.load_from_pretrained_model, map_location="cpu"
            )

            model.load_state_dict(state_dict)
            args.load_from_pretrained_model = None

        return model

    def max_positions(self) -> typing.OrderedDict[str, Tuple[int, int]]:
        return {
            key: (
                self.models[key].encoder.max_positions(),
                self.models[key].decoder.max_positions(),
            )
            for key in self.keys
        }


    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        prev_output_tokens: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_out = self.encoder(src_tokens, src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out, **kwargs)
        return decoder_out


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

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)


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

    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

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
