import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

from typing import Tuple


def get_data_loaders(data: Dataset | Tensor,
                     batch_size = 8192,
                     train_ratio= 0.7,
                     val_ratio  = 0.15,
                     test_ratio = 0.15,
                     num_workers=8
                     ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    assert train_ratio + val_ratio + test_ratio == 1
    n = data.shape[0]
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

