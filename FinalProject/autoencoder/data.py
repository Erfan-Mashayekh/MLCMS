import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.datasets import make_swiss_roll

from torch import Tensor
from typing import Tuple
from numpy import ndarray


def get_swiss_roll_data(n: int, normalize=True) -> Tuple[Tensor, ndarray]:
    np.random.seed(42)
    data, color = make_swiss_roll(n)
    data = torch.Tensor(data)
    if normalize:
        data = normalize_data(data)
    return data, color


class EmbeddingDataset(Dataset):
    """
    Is used to create datasets from dimensionality reduction techniques
    like Isomaps and Laplacian Eigenmaps (a.k.a. Spectral Embedding).
    Can be used to train Encoder to interpolate the mapping behavior s.t.
    it can be applied to new data.
    """
    def __init__(self, X: ndarray, Y: ndarray) -> None:
        """
        Args:
            X (ndarray, shape: (N, D)): data that is embedded
            Y (ndarray, shape: (N, D_latent)): resulting embedding
        """
        assert X.shape[0] == Y.shape[0]
        self.X = Tensor(X)
        self.Y = Tensor(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.Y[idx]


def get_data_loaders(data: Dataset | Tensor,
                     batch_size = 8192,
                     train_ratio= 0.7,
                     val_ratio  = 0.15,
                     test_ratio = 0.15,
                     num_workers=8,
                     normalize=True
                     ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation and training dataloader given a Tensor or Dataset

    Args:
        data (Dataset | Tensor): Complete data
        batch_size (int, optional): Defaults to 8192.
        train_ratio (float, optional): Defaults to 0.7.
        val_ratio (float, optional): Defaults to 0.15.
        test_ratio (float, optional): Defaults to 0.15.
        num_workers (int, optional): Number of subprocesses used for
            data-loading. Defaults to 8.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train, validation, test)
    """
    assert train_ratio + val_ratio + test_ratio == 1
    if type(data) == Tensor and normalize:
        data = normalize_data(data)
    n = len(data)
    n_train =   int(n * train_ratio)
    n_val =     int(n * val_ratio)
    n_test =    int(n * test_ratio)
    train_set, val_set, test_set = random_split(data, [n_train, n_val, n_test])
    train_loader= DataLoader(train_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)
    val_loader  = DataLoader(val_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)
    test_loader = DataLoader(test_set,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)
    return train_loader, val_loader, test_loader


def normalize_data(data) -> Tensor:
    data = Tensor(data)
    return (data - data.mean(dim=0)) / data.std(dim=0)
