from typing import Literal

import torch
from torch.types import _size
from scipy import signal

from score_inverse.tasks.task import DecomposeddSVDInverseTask


class DeblurTask(DecomposeddSVDInverseTask):
    def __init__(
        self,
        x_shape: _size,
        kernel_size: int = 5,
        kernel_type: Literal["uniform", "gaussian"] = "gaussian",
    ):
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type

        if kernel_type == "gaussian":
            std = 10
            kernel_1d = signal.windows.gaussian(kernel_size, std=std)
            kernel_1d = torch.from_numpy(kernel_1d)
        elif kernel_type == "uniform":
            kernel_1d = torch.ones(kernel_size)
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

        self.kernel_1d = kernel_1d / torch.sum(kernel_1d)

        super().__init__(x_shape)

    @staticmethod
    def linearize_kernel(kernel: torch.Tensor, input_size: int) -> torch.Tensor:
        A = torch.zeros(input_size - 2 * (kernel.shape[0] // 2), input_size)
        for i in range(A.shape[0]):
            for j in range(kernel.shape[0]):
                j_pos = i + j
                if 0 <= j_pos < A.shape[1]:
                    A[i, j_pos] = kernel[j]
        return A

    @property
    def A_row(self) -> torch.Tensor:
        return self.linearize_kernel(self.kernel_1d, self.x_shape[1])

    @property
    def A_col(self) -> torch.Tensor:
        return self.linearize_kernel(self.kernel_1d, self.x_shape[2])

    def noise(self, n: int) -> torch.Tensor:
        return torch.tensor(0)
