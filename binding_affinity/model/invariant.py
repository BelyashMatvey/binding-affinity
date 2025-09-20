import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing, global_mean_pool

from binding_affinity.data.types import AtomicInterfaceGraph


class InvariantLayer(MessagePassing):
    def __init__(self, edge_dim: int, node_dim: int, hidden_dim: int, aggr: str = "sum") -> None:
        super().__init__(aggr)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, h=h, edge_attr=edge_attr)

    # Ваше решение
    def message(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor) -> Tensor:
        tmp=torch.cat([h_i,h_j,edge_attr],dim=1)
        return self.message_mlp(tmp)


class RadialBasisExpansion(nn.Module):
    """
    Преобразует значения межатомных расстояний в вектор со значениями в [0, 1]
    с помощью набора радиальных базисных функций.
    """

    offset: Tensor

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 32,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class InvariantGNN(nn.Module):
    def __init__(
        self,
        node_vocab_size: int,  # кол-во типов вершин, например атомов
        node_dim: int,  # размерность эмбеддинга вершины
        edge_dim: int,  # размерность эмбеддинга ребра
        n_layers: int,  # кол-во графовых слоёв
        dropout: float = 0.0,  # dropout rate
    ) -> None:
        super().__init__()
        # этот эмбеддинг отвечает за тип атома
        self.node_embed = nn.Embedding(node_vocab_size, node_dim)
        # а этот — за класс ребра (внутри одного белка или между атомами разных белков)
        self.subunit_embed = nn.Embedding(2, embedding_dim=edge_dim // 2)
        # этот модуль разложит межатомные расстояния в вектор
        # похоже на разбивку по бинам, но более сглаженный вариант
        self.edge_rbf = RadialBasisExpansion(start=0, stop=5, num_gaussians=edge_dim // 2)
        # далее идёт последовательность наших графовых слоёв
        self.layers = nn.ModuleList(
            [InvariantLayer(edge_dim, node_dim, node_dim * 2, aggr="sum") for _ in range(n_layers)]
        )
        # полносвязный слой для отображения из эмбеддинга интерфейса в одно число
        self.fc = nn.Linear(node_dim, 1)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, batch: AtomicInterfaceGraph) -> Tensor:
        # эмбеддим типы атомов
        h = self.node_embed(batch.atoms)

        # считаем межатомные расстояния
        edge_index = batch.edge_index
        src, dst = edge_index
        distances = torch.linalg.norm(batch.coordinates[src] - batch.coordinates[dst], dim=1)

        # преобразуем межатомные расстояния в векторы с помощью RadialBasisExpansion
        edge_attr = self.edge_rbf(distances)

        # Сделаем с рёбрами более сложную вещь:
        # добавим информацию о типе ребра в его векторное представление

        # 1. определяем тип ребра (0 - ребро внутреннее, 1 — внешнее)
        is_intermolecular_edge = (batch.is_receptor[src]!=batch.is_receptor[dst]).long()
        # 2. эмбеддим тип ребра
        edge_kind_embed = self.subunit_embed(is_intermolecular_edge)
        # 3. эмбеддим рёбра
        edge_attr = torch.cat([edge_attr, edge_kind_embed], dim=1)

        # далее наши графовые слои + skip connections
        for layer in self.layers:
            h = (h + layer(h, edge_index, edge_attr)).relu()

        # сжимаем все эмбеддинги атомов в эмбеддинги интерфейса
        h = global_mean_pool(h,batch.batch)

        # dropout + linear поверх эмбеддинга графа
        h = self.dropout(h)
        h=self.fc(h)

        return h.squeeze(-1)
