from functools import partial

import torch
from torch.types import _size
from torchdrift.data.functional import gaussian_noise, impulse_noise, shot_noise

from score_inverse.tasks.task import DecomposeddSVDInverseTask


class DenoiseTask(DecomposeddSVDInverseTask):
    def __init__(self, x_shape: _size, noise_type: str = "normal", severity: int = 1):
        super().__init__(x_shape)

        assert len(x_shape) == 3
        self.c, self.h, self.w = x_shape

        if noise_type in ["gaussian", "normal"]:
            self.noiser = partial(gaussian_noise, severity=severity)
        elif noise_type in ["poisson", "shot"]:
            self.noiser = partial(shot_noise, severity=severity)
        elif noise_type in ["salt and pepper", "impulse"]:
            self.noiser = partial(impulse_noise, severity=severity)
        else:
            raise Exception("Invalid noise type")

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        return self.noiser(x)
