from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected

from binding_affinity.data.constants import ATOMS_INDICES, RESIDUE_INDICES
from binding_affinity.data.types import AtomicInterfaceGraph


def create_interface_graph(
    interface_structure: dict[str, Any],
    graph_radius: float = 4.0,
    n_neighbors: int = 10,
) -> AtomicInterfaceGraph:
    # преобразуем названия атомов в индексы
    encoded_atoms = torch.tensor(
        [ATOMS_INDICES.get(atom, len(ATOMS_INDICES)) for atom in interface_structure["atoms"]]
    )
    # то же для аминокислот
    encoded_residues = torch.tensor(
        [RESIDUE_INDICES.get(res, len(RESIDUE_INDICES)) for res in interface_structure["residues"]]
    )

    is_receptor = torch.tensor(interface_structure["is_receptor"])

    # тензор с координатами атомов
    coordinates = torch.tensor(interface_structure["coords"]).float()

    # используем координаты для построения радиус-графа:
    # NB: модели torch geometric обычно интерпретируют рёбра как направленные,
    # так что мы добавляем обратные рёбра с помощью функции `to_undirected`,
    # если хотим работать с неориентированными графами
    edge_index = to_undirected(
        radius_graph(coordinates, r=graph_radius, max_num_neighbors=n_neighbors)
    )
    # посчитаем расстояния
    src, tgt = edge_index
    distances = torch.linalg.norm(coordinates[src] - coordinates[tgt], dim=1)

    return Data(
        atoms=encoded_atoms,
        residues=encoded_residues,
        is_receptor=is_receptor,
        coordinates=coordinates,
        edge_index=edge_index,
        distances=distances,
        num_nodes=len(encoded_atoms),
    )
