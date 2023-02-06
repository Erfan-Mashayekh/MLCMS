import torch
import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, \
        ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from nnAutoencoder import Autoencoder
from nnWord2VecAE import Word2VecAutoencoder
from nnHybridAutoencoder import Encoder as HybridEncoder
from nnHybridAutoencoder import Decoder as HybridDecoder

from typing import Tuple, Dict


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG_DIR = os.path.join(os.getcwd(), "training_logs")
TENSORBOARD_LOG_DIR = os.path.join(os.getcwd(), "tensorboard/")


def train_embedding(train_loader: DataLoader,
                    val_loader:   DataLoader,
                    test_loader:  DataLoader,
                    latent_dim: int,
                    data_dim: int,
                    title = "Encoder"
                    ) -> Tuple[Autoencoder, Dict]:
    model = HybridEncoder(data_dim, latent_dim).to(DEVICE)
    return _train_swiss_roll(model, train_loader, val_loader, test_loader, title)



def train_reconstruction(train_loader: DataLoader,
                      val_loader:   DataLoader,
                      test_loader:  DataLoader,
                      latent_dim: int,
                      data_dim: int,
                      title = "Decoder"
                      ) -> Tuple[Autoencoder, Dict]:
    model = HybridDecoder(data_dim, latent_dim).to(DEVICE)
    return _train_swiss_roll(model, train_loader, val_loader, test_loader, title)


def train_autoencoder_swiss_roll(train_loader: DataLoader,
                      val_loader:   DataLoader,
                      test_loader:  DataLoader,
                      latent_dim: int,
                      data_dim: int,
                      title: str
                      ) -> Tuple[Autoencoder, Dict]:
    model = Autoencoder(data_dim, data_dim, latent_dim).to(DEVICE)
    return _train_swiss_roll(model, train_loader, val_loader, test_loader, title)


def train_word2vec_autoencoder(train_loader: DataLoader,
                                latent_dim: int,
                                data_dim: int,
                                title = "word2vec"
                                ) -> Tuple[Autoencoder, Dict]:
    model = Word2VecAutoencoder(data_dim, data_dim, latent_dim).to(DEVICE)
    return _train_word2vec(model, train_loader, title)


def _train_swiss_roll(model: pl.LightningModule,
                    train_loader: DataLoader,
                    val_loader:   DataLoader,
                    test_loader:  DataLoader,
                    title: str
                    ) -> Tuple[Autoencoder, Dict]:
    logger = TensorBoardLogger(TENSORBOARD_LOG_DIR + title)
    trainer = pl.Trainer(
        default_root_dir = os.path.join(LOG_DIR, title),
        accelerator='gpu' if str(DEVICE).startswith("cuda") else 'cpu',
        devices=1,
        max_epochs=1000,
        callbacks=[
            ModelCheckpoint(monitor="val_loss",
                            save_weights_only=True,
                            every_n_epochs=1),
            EarlyStopping(monitor="val_loss", patience=50, mode="min"),
            LearningRateMonitor("epoch")
        ],
        logger=logger,
        log_every_n_steps=1
    )

    trainer.logger._log_graph = True  # plot the computation graph in tb
    trainer.fit(model, train_loader, val_loader)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)

    result = {"val": val_result}
    return model, result


def _train_word2vec(model: pl.LightningModule,
           train_loader: DataLoader,
           title: str
           ) -> Tuple[Autoencoder, Dict]:
    logger = TensorBoardLogger(TENSORBOARD_LOG_DIR + title)
    trainer = pl.Trainer(
        default_root_dir = os.path.join(LOG_DIR, title),
        accelerator='gpu' if str(DEVICE).startswith("cuda") else 'cpu',
        devices=1,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(monitor="val_loss",
                            save_weights_only=True,
                            every_n_epochs=1),
            EarlyStopping(monitor="compression_loss", patience=3,
                          mode="min", min_delta=0.0015, verbose=False),
            LearningRateMonitor("epoch")
        ],
        logger=logger,
        log_every_n_steps=10
    )

    trainer.logger._log_graph = True  # plot the computation graph in tb
    trainer.fit(model, train_loader)

    # Test best model on validation and test set
    result = trainer.test(model, dataloaders=train_loader, verbose=False)

    result = {"result": result}
    return model, result

