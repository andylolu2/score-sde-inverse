import torch
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.transforms import ToTensor


class CIFAR10(Dataset):
    def __init__(self, is_training=False) -> None:
        super().__init__()

        self.dataset = _CIFAR10(
            "~/.cache/torchvision",
            download=True,
            transform=ToTensor(),
            train=is_training,
        )
        self.img_size = (3, 32, 32)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        x, y = self.dataset[index]
        return x
