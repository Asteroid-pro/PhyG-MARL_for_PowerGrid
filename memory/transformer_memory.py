"""
Transformer 时序记忆：保存连续时间步的图嵌入，提取暂态攻击和电网运行历史表示。
编辑时间：2026-08-24
作者：蔡景琪
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

from config import MEMORY_HISTORY_LEN, MEMORY_NHEAD, MEMORY_NUM_LAYERS


class TemporalMemory(nn.Module):
    def __init__(self, hidden_dim: int, history_length: int = MEMORY_HISTORY_LEN, num_layers: int = MEMORY_NUM_LAYERS, nhead: int = MEMORY_NHEAD):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_length = history_length
        self.buffer = deque(maxlen=history_length)
        self.positional_encoding = self._build_positional_encoding(history_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _build_positional_encoding(self, length: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def clear(self):
        self.buffer.clear()

    def update(self, new_graph_embedding: torch.Tensor):
        self.buffer.append(new_graph_embedding.detach())

    def forward_sequence(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected embeddings [batch, time, {self.hidden_dim}], got {tuple(embeddings.shape)}"
            )
        batch_size, sequence_length, _ = embeddings.shape
        sequence = torch.clamp(torch.nan_to_num(embeddings, nan=0.0), -10, 10)
        if sequence_length <= self.history_length:
            positions = self.positional_encoding[:sequence_length].to(sequence.device)
        else:
            positions = self._build_positional_encoding(sequence_length, self.hidden_dim).to(sequence.device)
        sequence = sequence + positions.unsqueeze(0)
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, sequence_length):
                raise ValueError("padding_mask must have shape [batch, time]")
            padding_mask = padding_mask.to(device=sequence.device, dtype=torch.bool)
        return self.encoder(sequence, src_key_padding_mask=padding_mask)

    def get_memory(self) -> torch.Tensor:
        if len(self.buffer) == 0:
            seq = torch.zeros(self.history_length, self.hidden_dim)
        else:
            seq_list = list(self.buffer)
            if len(seq_list) < self.history_length:
                pad = [torch.zeros(self.hidden_dim, device=seq_list[0].device) for _ in range(self.history_length - len(seq_list))]
                seq_list = pad + seq_list
            seq = torch.stack(seq_list, dim=0)
        return self.forward_sequence(seq.unsqueeze(0)).squeeze(0)[-1]


if __name__ == "__main__":
    mem = TemporalMemory(hidden_dim=16, history_length=4, num_layers=1, nhead=4)
    for _ in range(3):
        mem.update(torch.randn(16))
    m = mem.get_memory()
    assert m.shape == (16,)
    print("transformer_memory ok", m.shape)
