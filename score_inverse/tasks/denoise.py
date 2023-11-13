import torch
from torch.types import _size

from score_inverse.tasks.task import InverseTask


class DenoiseTask(InverseTask):
    def __init__(self, x_shape: _size, noise_std: float):
        super().__init__()

        assert len(x_shape) == 3
        self.c, self.h, self.w = x_shape
        self.noise_std = noise_std

    def noise(self, n: int) -> torch.Tensor:
        # TODO: Add option to add different kind of noise (e.g. shot noise)
        return self.noise_std * torch.randn((n, self.c, self.h, self.w))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def transform_inv(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def mask_inv(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def drop_inv(self, y: torch.Tensor) -> torch.Tensor:
        return y
