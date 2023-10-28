from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def get_dataset():
    return CIFAR10("~/.cache/torchvision", download=True, transform=ToTensor())
