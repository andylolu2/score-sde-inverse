import torch

from torchvision.datasets import CelebA as _CelebA
from torchvision.transforms import ToTensor, Resize, Compose, CenterCrop
from torch.utils.data import Dataset


class CelebA(Dataset):
    def __init__(self, img_size: int = 64) -> None:
        super().__init__()

        self.dataset = _CelebA(
            "~/.cache/torchvision",
            download=True,
            transform=Compose(
                [
                    CenterCrop(140),
                    Resize(img_size, antialias=True),
                    ToTensor(),
                ]
            ),
        )
        self.img_size = (3, img_size, img_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        x, y = self.dataset[index]
        return x
