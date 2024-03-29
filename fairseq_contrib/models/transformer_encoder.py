import math
import typing
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import FairseqEncoder
from fairseq.modules import LayerNorm, LearnedPositionalEmbedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq_contrib.data import MaskedDictionary
from .transformer_decoder import DecoderOutput
from .transformer_encoder_layer import TransformerEncoderLayer

TensorDict = Dict[str, Optional[torch.Tensor]]


class TransformerEncoder(FairseqEncoder):
    def __init__(
        self,
        dictionary: MaskedDictionary,
        dropout: float,
        max_source_positions: int,
        num_layers: int,
        embedding_languages: torch.nn.Embedding,
        embedding_tokens: torch.nn.Embedding,
        encoder_ffn_embedding_dim: int,
        encoder_num_attention_heads: int,
        encoder_attention_dropout: float,
        encoder_activation_dropout: float,
        encoder_activation_fn: str,
        use_token_positions: bool = False,
    ) -> None:
        super().__init__(dictionary)

        self.register_buffer("version", torch.Tensor([4]))

        self.pad_index = dictionary.pad()

        encoder_embedding_dim = embedding_tokens.embedding_dim

        self.dropout = dropout
        self.max_source_positions = max_source_positions
        self.embedding_languages = embedding_languages
        self.embedding_tokens = embedding_tokens
        self.embedding_scale = math.sqrt(encoder_embedding_dim)
        self.use_token_positions = use_token_positions

        if use_token_positions:
            self.embedding_positions = LearnedPositionalEmbedding(
                num_embeddings=max_source_positions + 1 + self.pad_index,
                embedding_dim=encoder_embedding_dim,
                padding_idx=self.pad_index,
            )
        else:
            self.embedding_positions = LearnedPositionalEmbedding(
                num_embeddings=max_source_positions,
                embedding_dim=encoder_embedding_dim,
                padding_idx=None,
            )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    encoder_embedding_dim,
                    encoder_ffn_embedding_dim,
                    encoder_num_attention_heads,
                    dropout,
                    encoder_attention_dropout,
                    encoder_activation_dropout,
                    encoder_activation_fn,
                )
                for _ in range(num_layers)
            ]
        )

        self.embedding_layer_norm = LayerNorm(encoder_embedding_dim)
        self.apply(init_bert_params)

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor,
        **kwargs: typing.Any,
    ) -> Dict[str, torch.Tensor]:
        pos_tokens = None  # src_tokens[:, kwargs["positions"]]
        print(f"[Encoder2] tokens={src_tokens}")
        # languages = kwargs["languages"]
        # print(f"[EnCODER] lan={languages}")
        # shape: [Batch, Time]
        encoder_padding_mask = src_tokens == self.pad_index
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # prepare source positions
        if self.use_token_positions:
            positions = self.embedding_positions(src_tokens)
        else:
            src_len = src_tokens.size(1)
            positions = torch.arange(src_len).unsqueeze(0)
            positions = positions.to(src_tokens.device)
            # shape: [Batch=1, Time, Channel]
            positions = self.embedding_positions(None, positions=positions)

        # kwargs["languages"] = (
        #     torch.LongTensor([0]).expand_as(src_tokens).to(src_tokens.device)
        # )

        if "languages" in kwargs:
            # shape: [Batch, Time, Channel]
            languages = self.embedding_languages(kwargs["languages"])
        else:
            languages = None

        # shape: [Batch, Time, Channel]
        x = self.embedding_scale * self.embedding_tokens(src_tokens)
        # add positions embeddings
        x += positions.expand_as(x)
        # add languages embeddings
        if languages is not None:
            x += languages
        x = self.embedding_layer_norm(x)
        x = F.dropout(x, self.dropout, self.training)

        if encoder_padding_mask is not None:
            # shape: [Batch, Time, Channel]
            encoder_padding_mask = encoder_padding_mask.unsqueeze(-1)
            x *= 1 - encoder_padding_mask.type_as(x)

        # shape: [Time, Batch, Channel]
        x = x.transpose(0, 1)

        for layer in self.layers:
            # shape: [Batch, Time]
            if encoder_padding_mask is not None:
                encoder_padding_mask = encoder_padding_mask.squeeze(-1)
            x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask)

        return {
            "encoder_out": x,
            "encoder_padding_mask": encoder_padding_mask,
        }

    def reorder_encoder_out(
        self, encoder_out: Dict[str, torch.Tensor], new_order: torch.LongTensor
    ) -> Dict[str, torch.Tensor]:
        # reorder `Batch` dimension according to `new_order`
        if encoder_out["encoder_out"] is not None:
            # shape: [Time, Batch, Channel]
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            # shape: [Batch, Time]
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self) -> int:
        if self.embedding_positions is None:
            return self.max_source_positions
        return min(
            self.max_source_positions, self.embedding_positions.max_positions
        )

    def get_normalized_probs(
        self,
        net_output: DecoderOutput,
        log_probs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError
