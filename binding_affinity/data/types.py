from __future__ import annotations

from typing import Protocol

from torch import Tensor


class AtomicInterfaceGraph(Protocol):
    atoms: Tensor  # (N): идентификаторы типов атомов
    residues: Tensor  # (N): идентификаторы аминокислот
    is_receptor: Tensor  # (N): 1 для атомов рецептора, 0 для атомов лиганда
    coordinates: Tensor  # (N x 3): координаты атомов
    edge_index: Tensor  # (2 x E) список рёбер между атомами
    distances: Tensor  # (E): расстояния между атомами, соединёнными ребром
    affinity: Tensor | None  # (n): свободная энергия связывания
    batch: Tensor | None  # (N): идентификаторы подграфов в батче, [0, n-1]

    def clone(self) -> AtomicInterfaceGraph: ...
