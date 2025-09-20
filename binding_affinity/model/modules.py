from typing import Type

from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import global_mean_pool

from binding_affinity.data.constants import ATOMS_INDICES
from binding_affinity.data.types import AtomicInterfaceGraph


class GraphNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        graph_layer: Type[MessagePassing],
        n_layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        # эмбеддинг для типов атомов
        self.embed = nn.Embedding(len(ATOMS_INDICES) + 1, hidden_dim)
        # список графовых слоёв
        self.conv = nn.ModuleList([graph_layer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        # линейный слой для регрессии
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, batch: AtomicInterfaceGraph):
        # 1. Эмбеддинги вершин
        x = self.embed(batch.atoms)
        for conv in self.conv:
            x = (x + conv(x, batch.edge_index)).relu()

        # 2. Эмбеддинг графа: усреднение по вершинам отдельных графов
        x = global_mean_pool(x, batch.batch)  # [batch_size, hidden_channels]

        # 3. Финальный регрессор поверх эмбеддинга графа
        x = self.dropout(x)
        x = self.fc(x)
        return x
