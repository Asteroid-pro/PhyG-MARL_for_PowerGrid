from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data


IEEE14_LINE_DATA = [
    (0, 1, 0.01938, 0.05917),
    (0, 4, 0.05403, 0.22304),
    (1, 2, 0.04699, 0.19797),
    (1, 3, 0.05811, 0.17632),
    (1, 4, 0.05695, 0.17388),
    (2, 3, 0.06701, 0.17103),
    (3, 4, 0.01335, 0.04211),
    (3, 6, 0.0, 0.20912),
    (3, 8, 0.0, 0.55618),
    (4, 5, 0.0, 0.25202),
    (5, 10, 0.09498, 0.19890),
    (5, 11, 0.12291, 0.25581),
    (5, 12, 0.06615, 0.13027),
    (6, 7, 0.0, 0.17615),
    (6, 8, 0.0, 0.11001),
    (8, 9, 0.03181, 0.08450),
    (8, 13, 0.12711, 0.27038),
    (9, 10, 0.08205, 0.19207),
    (11, 12, 0.22092, 0.19988),
]


@dataclass
class LineRecord:
    src: int
    dst: int
    r: float
    x: float
    active: bool = True

    @property
    def admittance(self) -> float:
        denom = self.r * self.r + self.x * self.x
        return 0.0 if denom == 0 else float(1.0 / np.sqrt(denom))


class TopologyManager:
    def __init__(self, n_bus: int = 14):
        self.n_bus = 14
        self.lines = [LineRecord(*line) for line in IEEE14_LINE_DATA]
        self.adj_matrix = np.zeros((self.n_bus, self.n_bus), dtype=np.float32)
        self._rebuild_adj()

    def _rebuild_adj(self) -> None:
        self.adj_matrix.fill(0.0)
        for line in self.lines:
            if line.active:
                self.adj_matrix[line.src, line.dst] = 1.0
                self.adj_matrix[line.dst, line.src] = 1.0

    def add_edge(self, line_id: int) -> None:
        self.restore_edge(line_id)

    def remove_edge(self, line_id: int) -> None:
        self.lines[line_id].active = False
        self._rebuild_adj()

    def restore_edge(self, line_id: int) -> None:
        self.lines[line_id].active = True
        self._rebuild_adj()

    def get_disconnected_lines(self):
        return [idx for idx, line in enumerate(self.lines) if not line.active]

    def would_disconnect_grid(self, candidate_line_id: int) -> bool:
        adjacency = [[] for _ in range(self.n_bus)]
        for line_id, line in enumerate(self.lines):
            if line.active and line_id != candidate_line_id:
                adjacency[line.src].append(line.dst)
                adjacency[line.dst].append(line.src)
        visited = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return len(visited) != self.n_bus

    def get_non_islanding_active_lines(self) -> list[int]:
        return [
            line_id
            for line_id, line in enumerate(self.lines)
            if line.active and not self.would_disconnect_grid(line_id)
        ]

    def get_graph(self) -> Data:
        edge_index = []
        edge_attr = []
        for line in self.lines:
            if line.active:
                edge_index.append([line.src, line.dst])
                edge_index.append([line.dst, line.src])
                edge_attr.append([line.admittance])
                edge_attr.append([line.admittance])
        if edge_index:
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
            edge_attr_tensor = torch.empty((0, 1), dtype=torch.float32)
        return Data(edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, num_nodes=self.n_bus)


if __name__ == "__main__":
    topo = TopologyManager(14)
    topo.remove_edge(0)
    graph = topo.get_graph()
    assert graph.num_nodes == 14
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 1
    print("topology_manager ok", graph.edge_index.shape, graph.edge_attr.shape)
