import torch
from torch.types import _size

from score_inverse.tasks.task import DecomposeddSVDInverseTask


class SuperResolutionTask(DecomposeddSVDInverseTask):
    def __init__(self, x_shape: _size, scale_factor: int = 2):
        self.scale_factor = scale_factor
        super().__init__(x_shape)

    @property
    def A_row(self) -> torch.Tensor:
        A = torch.zeros(self.x_shape[1] // self.scale_factor, self.x_shape[1])
        for i in range(self.x_shape[1] // self.scale_factor):
            A[i, i * self.scale_factor : (i + 1) * self.scale_factor] = (
                1 / self.scale_factor
            )
        return A

    @property
    def A_col(self) -> torch.Tensor:
        A = torch.zeros(self.x_shape[2] // self.scale_factor, self.x_shape[2])
        for i in range(self.x_shape[2] // self.scale_factor):
            A[i, i * self.scale_factor : (i + 1) * self.scale_factor] = (
                1 / self.scale_factor
            )
        return A

    def noise(self, n: int) -> torch.Tensor:
        c, h, w = self.x_shape
        return torch.zeros(n, c, h // self.scale_factor, w // self.scale_factor)
