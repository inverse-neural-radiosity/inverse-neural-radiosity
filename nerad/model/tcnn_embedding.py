from typing import Any

import tinycudann as tcnn
import torch
import torch.nn as nn


class TcnnEmbedding(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.embedding = tcnn.Encoding(3, config, dtype=torch.float32)
        self.n_output_dims = self.embedding.n_output_dims

    def forward(self, x):
        return self.embedding.forward(x)
