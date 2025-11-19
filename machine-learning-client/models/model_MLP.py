"""
Model for predicting letter
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    MLP block: x -> LN -> Linear -> GELU -> Dropout -> Linear -> +x
    """
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.3):
        super().__init__()
        hidden = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class LandmarkMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 24,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, expansion=2, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        # Final head
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 63) normalized landmark vectors
        """
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = F.gelu(x)

        for block in self.blocks:
            x = block(x)

        x = self.head_norm(x)
        logits = self.head(x)
        return logits
