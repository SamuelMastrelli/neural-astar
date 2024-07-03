from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar
from neural_astar.utils.data_sdd_interscenes import create_sdd_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint 
import sys

@hydra.main(config_path="config", config_name="train_sdd_inter")
def main(config):

    set_global_seeds(config.seed)
    train_loader1 = create_sdd_dataloader(config.dataset, ["quad", "nexus", "little", "hyang", "gates", "deathCircle", "coupa"], config.params.batch_size)
    val_loader1 = create_sdd_dataloader(config.dataset, ["bookstore"], config.params.batch_size)

    neural_astar = NeuralAstar(   
        encoder_arch=config.encoder.arch,
        encoder_depth=config.encoder.depth,
        encoder_input=config.encoder.input,
        const=config.encoder.const,
        learn_obstacles=True,
        Tmax=config.Tmax,
    )
     
    im, s, g, t = next(iter(train_loader1))
    print(im.shape, s.shape, g.shape, t.shape)

    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/val_loss", save_weights_only=True, mode="max"
    )

    module = PlannerModule(neural_astar, config)
    logdir= f"{config.logdir}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, train_loader1, val_loader1)


if __name__ == "__main__":
    main()