import hydra
import lightning as L
from lightning.pytorch import Trainer
from omegaconf import DictConfig
from binding_affinity.data.datamodule import DataModule
from binding_affinity.model.lightning import AffinityModel

@hydra.main(version_base=None, config_path="../config", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    callbacks = (
        [hydra.utils.instantiate(cb_conf) for _, cb_conf in cfg.callbacks.items()]
        if cfg.callbacks
        else []
    )
    logger = hydra.utils.instantiate(cfg.logger)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)
    lit: AffinityModel = hydra.utils.instantiate(cfg.lightning)
    datamodule: DataModule = hydra.utils.instantiate(cfg.datamodule)

    trainer.fit(
        model=lit,
        datamodule=datamodule,
        ckpt_path=cfg.checkpoint,
    )


if __name__ == "__main__":
    train()
