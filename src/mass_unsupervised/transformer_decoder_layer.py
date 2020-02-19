import typing
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.utils import get_activation_fn

TensorDict = Dict[str, Optional[torch.Tensor]]


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        encoder_embedding_dim: int = 768,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
    ) -> None:
        super().__init__()

        # basic blocks
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)

        # self attention block
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        # encoder attention block
        self.encoder_attn = MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            kdim=encoder_embedding_dim,
            vdim=encoder_embedding_dim,
            dropout=attention_dropout,
            encoder_decoder_attention=True,
        )

        # layer norm associated with the encoder attention layer
        self.encoder_attn_layer_norm = LayerNorm(
            self.embedding_dim, export=export
        )

        # Feed-forward layers
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.need_attn = False

    def forward(
        self,
        x: torch.Tensor,
        encoder_out=torch.Tensor,
        encoder_padding_mask: Optional[torch.ByteTensor] = None,
        incremental_state: Optional[Dict[str, TensorDict]] = None,
        prev_self_attn_state: Optional[torch.Tensor] = None,
        prev_attn_state: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.ByteTensor] = None,
        self_attn_padding_mask: Optional[torch.ByteTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pass incremental state to decoder attention
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state: TensorDict = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }

            self.self_attn._set_input_buffer(incremental_state, saved_state)

        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        x = F.dropout(x, self.dropout, self.training)
        x += residual
        x = self.self_attn_layer_norm(x)

        # pass incremental state to encoder attention
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state
            saved_state: TensorDict = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }

            self.encoder_attn._set_input_buffer(incremental_state, saved_state)

        need_encoder_attn_weights = False
        if not self.training:
            if self.need_attn:
                need_encoder_attn_weights = True

        residual = x
        # TODO: make encoder_attn optional
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=need_encoder_attn_weights,
        )

        x = F.dropout(x, self.dropout, self.training)
        x += residual
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, self.activation_dropout, self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout, self.training)
        x += residual
        x = self.final_layer_norm(x)

        return x, attn

    def make_generation_fast(
        self, need_attn: bool = False, **kwargs: typing.Any
    ) -> None:
        self.need_attn = need_attn
