from pathlib import Path

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch_geometric.loader import DataLoader

from binding_affinity.data.dataset import AtomicGraphDataset


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        datadir: Path,
        batch_size: int,
        graph_radius: int = 4,
        n_neighbors: int = 10,
        bipartite: bool = True,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.datadir = Path(datadir)
        self.graph_radius = graph_radius
        self.n_neighbors = n_neighbors
        self.bipartite = bipartite

    def setup(self, stage: str) -> None:
        if stage in ("fit", "validate"):
            self.val_dataset = AtomicGraphDataset(
                self.datadir / "affinity_test.json",
                self.graph_radius,
                self.n_neighbors,
                self.bipartite,
            )
        if stage == "fit":
            self.train_dataset = AtomicGraphDataset(
                self.datadir / "affinity_train.json",
                self.graph_radius,
                self.n_neighbors,
                self.bipartite,
            )
        elif stage == "test":
            self.test_dataset = AtomicGraphDataset(
                self.datadir / "affinity_test.json",
                self.graph_radius,
                self.n_neighbors,
                self.bipartite,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
