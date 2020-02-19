import torch
import torch.nn as nn
from typing import Optional


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # create an embedding layer
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        # initialize weights with He et. al initialization method
        nn.init.normal_(
            self.embedding.weight, mean=0, std=embedding_dim ** -0.5
        )
        # skip padding elements
        nn.init.constant_(self.embedding.weight[padding_idx], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        # create a fully connected layer
        self.fc = nn.Linear(in_features, out_features, bias)
        # initialize weights with Xavier et al initialization method
        nn.init.xavier_uniform_(self.fc.weight)
        if bias:
            # fill bias with zeroes
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
