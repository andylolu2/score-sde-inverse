import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA as _CelebA
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor


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
            split="test",
        )
        self.img_size = (3, img_size, img_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        x, y = self.dataset[index]
        return x
