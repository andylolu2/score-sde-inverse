import torch
from torch.types import _size

from score_inverse.tasks.task import DecomposeddSVDInverseTask


class DenoiseTask(DecomposeddSVDInverseTask):
    def __init__(self, x_shape: _size, noise_std: float):
        super().__init__(x_shape)
        self.noise_std = noise_std

    def noise(self, n: int) -> torch.Tensor:
        # TODO: Add option to add different kind of noise (e.g. shot noise)
        return self.noise_std * torch.randn((n, *self.output_shape))
