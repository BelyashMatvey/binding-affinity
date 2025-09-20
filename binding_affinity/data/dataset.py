import json
from pathlib import Path

from torch.utils.data import Dataset
from torch_geometric.transforms.remove_isolated_nodes import RemoveIsolatedNodes

from binding_affinity.data.transforms import create_interface_graph
from binding_affinity.data.types import AtomicInterfaceGraph


class AtomicGraphDataset(Dataset):
    def __init__(
        self,
        data_json: Path,
        graph_radius: float = 4.0,
        n_neighbors: int = 10,
        bipartite: bool = True,
    ) -> None:
        self.data: list[AtomicInterfaceGraph] = []
        for x in json.loads(data_json.read_text()):
            item = create_interface_graph(x["interface_graph"], graph_radius, n_neighbors)
            item.affinity = x["affinity"]
            # удаление внутренних рёбер
            if bipartite:
                item = self.remove_intermolecular_edges(item)
            self.data.append(item)

    def __getitem__(self, index: int) -> AtomicInterfaceGraph:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def remove_intermolecular_edges(
        interface_graph: AtomicInterfaceGraph,
    ) -> AtomicInterfaceGraph:
        interface_bigraph = interface_graph.clone()
        src, tgt = interface_bigraph.edge_index
        intermolecular_edges = (
            (interface_graph.is_receptor[src] - interface_graph.is_receptor[tgt]).abs().bool()
        )
        interface_bigraph.edge_index = interface_bigraph.edge_index[:, intermolecular_edges]
        interface_bigraph.distances = interface_bigraph.distances[intermolecular_edges]

        interface_bigraph = RemoveIsolatedNodes().forward(interface_bigraph)
        return interface_bigraph
