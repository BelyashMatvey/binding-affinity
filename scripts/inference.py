import hydra
from lightning.pytorch import Trainer
from omegaconf import DictConfig

import torch.nn.functional as F
from binding_affinity.data.datamodule import DataModule
from binding_affinity.model.lightning import AffinityModel

@hydra.main(version_base=None, config_path="../config", config_name="inference_invariant.yaml")
def inference(cfg: DictConfig) -> None:
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=False)
    lit: AffinityModel = hydra.utils.instantiate(cfg.lightning)
    datamodule: DataModule = hydra.utils.instantiate(cfg.datamodule)

    trainer.test(
        model=lit,
        datamodule=datamodule,
        ckpt_path=cfg.checkpoint,
    )


if __name__ == "__main__":
    inference()
