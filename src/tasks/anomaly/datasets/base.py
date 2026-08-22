from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseAnomalyDataset(Dataset, ABC):
    """Base dataset class for anomaly detection tasks."""

    def __init__(self, root, split, transform=None, **params):
        self.root = root
        self.split = split
        self.transform = transform
        self.classes = ["normal", "anomalous"]

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
