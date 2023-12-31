import numpy as np
from typing import Any, Tuple
from torchvision.datasets import DatasetFolder


def load_npz_frames(file_name: str) -> np.ndarray:

    return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)


class DatasetForELM(DatasetFolder):
    """
    Custom dataset class for ELM (Extreme Learning Machine) model.

    Args:
        root (str): Root directory of the dataset. The directory should in the format of torchvision.datasets.DatasetFolder.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version.

    Attributes:
        root (str): Root directory of the dataset.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version.

    """

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.loader = load_npz_frames
        self.transform = transform
        self.target_transform = target_transform

        super(DatasetForELM, self).__init__(
            root=self.root,
            loader=self.loader, 
            extensions=('.npz',),
            transform=self.transform, 
            target_transform=self.target_transform
            )
    
    def __getitem__(self, index: int) -> Tuple[Any, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is the name of the target class.
        """
        path, class_index = self.samples[index]
        sample = self.loader(path)
        target = self.classes[class_index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target