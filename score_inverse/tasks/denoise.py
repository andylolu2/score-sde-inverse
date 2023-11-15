import torch
from torch.types import _size
from functools import partial

from torchdrift.data.functional import gaussian_noise, shot_noise, impulse_noise

from score_inverse.tasks.task import InverseTask

class DenoiseTask(InverseTask):
    def __init__(self, x_shape: _size, noise_type: str = 'normal', severity: int = 1):
        super().__init__()

        assert len(x_shape) == 3
        self.c, self.h, self.w = x_shape

        if noise_type in ['gaussian', 'normal']:
            self.noiser = partial(gaussian_noise, severity=severity)
        elif noise_type in ['poisson', 'shot']:
            self.noiser = partial(shot_noise, severity=severity)
        elif noise_type in ['salt and pepper', 'impulse']:
            self.noiser = partial(impulse_noise, severity=severity)
        else:
            raise Exception('Invalid noise type')

    def noise(self, x: torch.Tensor) -> torch.Tensor:
        return self.noiser(x) - x


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
