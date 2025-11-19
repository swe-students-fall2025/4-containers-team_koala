from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class LandmarkMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 63,  # 21 landmarks * 3 coordinates
        num_classes: int = 24,
        hidden_dim: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, input_dim)
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        logits = self.fc_out(x)
        return logits
