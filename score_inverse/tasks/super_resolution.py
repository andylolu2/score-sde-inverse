import torch
from torch.types import _size

from score_inverse.tasks.task import DecomposeddSVDInverseTask


class SuperResolutionTask(DecomposeddSVDInverseTask):
    def __init__(self, x_shape: _size, scale_factor: int = 2):
        self.scale_factor = scale_factor
        super().__init__(x_shape)

    @property
    def bicubic_kernel(self):
        def k(x):
            a = -0.5
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        kernel = torch.zeros(self.scale_factor * 4)
        for i in range(self.scale_factor * 4):
            x = (1 / self.scale_factor) * (i - self.scale_factor * 2 + 0.5)
            kernel[i] = k(x)
        kernel = kernel / torch.sum(kernel)
        return kernel

    @property
    def A_row(self) -> torch.Tensor:
        s = self.scale_factor
        h = self.x_shape[1]
        A = torch.zeros(h // s, h)
        kernel = self.bicubic_kernel
        k = len(kernel)

        for i in range(len(A)):
            i_effective = s * i + s // 2  # index in original image
            for j in range(k):
                j_effective = i_effective + j - k // 2  # index in original image
                # reflective padding
                if j_effective < 0:
                    j_effective = -j_effective - 1
                if j_effective >= h:
                    j_effective = 2 * h - j_effective - 1
                A[i, j_effective] += kernel[j]

        return A

    @property
    def A_col(self) -> torch.Tensor:
        s = self.scale_factor
        w = self.x_shape[2]
        A = torch.zeros(w // s, w)
        kernel = self.bicubic_kernel
        k = len(kernel)

        for i in range(len(A)):
            i_effective = s * i + s // 2
            for j in range(k):
                j_effective = i_effective + j - k // 2
                # reflective padding
                if j_effective < 0:
                    j_effective = -j_effective - 1
                if j_effective >= w:
                    j_effective = 2 * w - j_effective - 1
                A[i, j_effective] += kernel[j]

        return A
