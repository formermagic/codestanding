import math
import typing
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import LayerNorm, LearnedPositionalEmbedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import fill_with_neg_inf, log_softmax, softmax

from fairseq_contrib.data import MaskedDictionary
from .transformer_decoder_layer import TransformerDecoderLayer

TensorDict = Dict[str, Optional[torch.Tensor]]
Extras = Dict[str, Union[torch.Tensor, List[Optional[torch.Tensor]]]]
DecoderOutput = Tuple[torch.Tensor, Extras]


class TransformerDecoder(FairseqIncrementalDecoder):
    def __init__(
        self,
        dictionary: MaskedDictionary,
        dropout: float,
        max_target_positions: int,
        num_layers: int,
        embedding_languages: torch.nn.Embedding,
        embedding_tokens: torch.nn.Embedding,
        encoder_embedding_dim: int,
        decoder_ffn_embed_dim: int,
        decoder_num_attention_heads: int,
        decoder_attention_dropout: float,
        decoder_activation_dropout: float,
        decoder_activation_fn: str,
        share_decoder_input_output_embeddings: bool,
        need_encoder_attn: bool = False,
    ) -> None:
        super().__init__(dictionary)

        self.register_buffer("version", torch.Tensor([4]))

        self._future_mask: Optional[torch.Tensor] = None

        self.dropout = dropout
        self.share_decoder_input_output_embeddings = (
            share_decoder_input_output_embeddings
        )

        # input_embedding_dim = embedding_tokens.embedding_tokens

        self.padding_index = dictionary.pad()
        self.max_target_positions = max_target_positions
        self.embedding_dim = embedding_tokens.embedding_dim
        self.embedding_languages = embedding_languages
        self.embedding_tokens = embedding_tokens
        self.embedding_scale = math.sqrt(self.embedding_dim)

        self.embedding_positions = LearnedPositionalEmbedding(
            self.max_target_positions + 1 + self.padding_index,
            self.embedding_dim,
            self.padding_index,
        )

        langs = max(embedding_languages.num_embeddings, 1)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    encoder_embedding_dim,
                    self.embedding_dim,
                    decoder_ffn_embed_dim,
                    decoder_num_attention_heads,
                    dropout,
                    decoder_attention_dropout,
                    decoder_activation_dropout,
                    decoder_activation_fn,
                    langs=langs,
                )
                for _ in range(num_layers)
            ]
        )

        if not share_decoder_input_output_embeddings:
            self.embedding_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.embedding_dim)
            )
            nn.init.normal_(
                self.embedding_out, mean=0, std=self.embedding_dim ** -0.5
            )

        self.embedding_layer_norm = LayerNorm(self.embedding_dim)
        self.apply(init_bert_params)

    def forward(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: Optional[Dict[str, torch.Tensor]] = None,
        incremental_state: Optional[Dict[str, TensorDict]] = None,
        **kwargs: typing.Any,
    ) -> DecoderOutput:
        # print(f"[Decoder] prev_output_tokens=..., kwargs={kwargs}")
        x, extras = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state, **kwargs
        )
        x = self.output_layer(x)
        return x, extras

    def extract_features(
        self,
        prev_output_tokens: torch.Tensor,
        encoder_out: Optional[Dict[str, torch.Tensor]] = None,
        incremental_state: Optional[Dict[str, TensorDict]] = None,
        **kwargs: typing.Any,
    ) -> DecoderOutput:
        if "positions" in kwargs:
            # shape: [Batch, Time, Channel]
            positions = self.embedding_positions(kwargs["positions"])
        else:
            # shape: [Batch, Time, Channel]
            positions = self.embedding_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        # kwargs["languages"] = (
        #     torch.LongTensor([0])
        #     .expand_as(prev_output_tokens)
        #     .to(prev_output_tokens.device)
        # )

        if "languages" in kwargs:
            # shape: [Batch, Time, Channel]
            languages = self.embedding_languages(kwargs["languages"])
            languages = languages[:, : prev_output_tokens.size(1), :]
            language = kwargs["languages"].max().cpu().item()
        else:
            languages = None
            language = 0

        if incremental_state is not None:
            # shape: [Batch, Time]
            prev_output_tokens = prev_output_tokens[:, -1:]

            if positions is not None:
                # shape: [Batch, Time, Channel]
                positions = positions[:, -1:]

            if languages is not None:
                # shape: [Batch, Time, Channel]
                languages = languages[:, -1:]

        # shape: [Batch, Time, Channel]
        x = self.embedding_scale * self.embedding_tokens(prev_output_tokens)

        if positions is not None:
            x += positions
        if languages is not None:
            x += languages

        x = self.embedding_layer_norm(x)
        x = F.dropout(x, self.dropout, self.training)

        # shape: [Time, Batch, Channel]
        x = x.transpose(0, 1)

        attn: Optional[torch.Tensor] = None
        inner_states: List[torch.Tensor] = [x]

        for layer in self.layers:
            encoder_out_tensor: Optional[torch.Tensor] = None
            if encoder_out is not None:
                encoder_out_tensor = encoder_out.get("encoder_out", None)

            encoder_padding_mask: Optional[torch.ByteTensor] = None
            if encoder_out is not None:
                encoder_padding_mask = encoder_out.get(
                    "encoder_padding_mask", None
                )

            self_attn_mask: Optional[torch.ByteTensor] = None
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)

            x, attn = layer(
                x,
                language,
                encoder_out=encoder_out_tensor,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
            )

            inner_states.append(x)

        # shape: [Batch, Time, Channel]
        x = x.transpose(0, 1)

        extras = {"attn": attn, "inner_states": inner_states}

        return x, extras

    def output_layer(
        self, features: torch.Tensor, **kwargs: typing.Any
    ) -> torch.Tensor:
        if self.share_decoder_input_output_embeddings:
            # shape: [Batch, Time, Channel]
            # Channel=len(target_dict)
            x = F.linear(features, self.embedding_tokens.weight)
        else:
            # shape: [Batch, Time, Channel]
            # Channel=len(target_dict)
            x = F.linear(features, self.embedding_out)

        return x

    def buffered_future_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def max_positions(self) -> int:
        if self.embedding_tokens is None:
            return self.max_target_positions
        return min(
            self.max_target_positions, self.embedding_positions.max_positions
        )

    def get_normalized_probs(
        self,
        net_output: DecoderOutput,
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        logits = net_output[0]
        if log_probs:
            return log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
