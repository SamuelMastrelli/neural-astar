"""
Training neural A * with maps dataset from 
https://github.com/micheleantonazzi/gibson-env-utilities
"""

from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl 
import torch 
from neural_astar.planner import NeuralAstar
from neural_astar.utils.data_maps import create_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import os


@hydra.main(config_path="config", config_name="train_maps")
def main(config):
    torch.cuda.empty_cache()
    set_global_seeds(config.seed)
    train_loader = create_dataloader(dir=config.dataset, cluster="train_resized", batch_size=config.params.batch_size)
    val_loader = create_dataloader(dir=config.dataset, cluster="validation_resized", batch_size=config.params.batch_size)

    m, s, g, t, _= next(iter(train_loader))
    print(m.shape, s.shape, g.shape, t.shape)


  

    neural_astar = NeuralAstar(
        encoder_arch=config.encoder.arch,
        encoder_depth=config.encoder.depth,
        encoder_input=config.encoder.input,
        learn_obstacles=False,
        Tmax=config.Tmax,
    )

 


    checkpoint_callbacks = ModelCheckpoint(
        monitor="metrics/h_mean", save_weights_only=False, mode="max"
    )

    module = PlannerModule(neural_astar, config, True)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator= "gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callbacks],
    )
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()
