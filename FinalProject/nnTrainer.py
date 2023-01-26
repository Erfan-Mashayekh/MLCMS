import torch
import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nnAutoencoder import Autoencoder

from typing import Tuple, Dict


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_DIR = os.path.join(os.getcwd(), "training_logs")
TENSORBOARD_LOG_DIR = os.path.join(os.getcwd(), "tensorboard")


def train_autoencoder(train_loader: DataLoader,
                      val_loader:   DataLoader,
                      test_loader:  DataLoader,
                      latent_dim: int,
                      data_dim: int
                      ) -> Tuple[pl.LightningModule, Dict]:
    logger = TensorBoardLogger(TENSORBOARD_LOG_DIR)
    trainer = pl.Trainer(
        default_root_dir = os.path.join(LOG_DIR, "Swiss_Role_%i" % latent_dim),
        accelerator='gpu' if str(DEVICE).startswith("cuda") else 0,
        devices=1 if str(DEVICE).startswith("cuda") else 0,
        max_epochs=100,
        callbacks=[
            ModelCheckpoint(save_weights_only=True),
             # TODO: GenerateCallback(every_n_epochs=10),
            LearningRateMonitor("epoch")
        ],
        logger=logger,
        log_every_n_steps=3
    )

    trainer.logger._log_graph = True  # plot the computation graph in tb

    model = Autoencoder(data_dim, data_dim, latent_dim=latent_dim)
    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


class GenerateCallback(pl.Callback):
    pass #TODO:/*