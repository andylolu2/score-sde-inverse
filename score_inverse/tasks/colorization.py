import torch
import torch.nn.functional as F
import torch.distributions as D
from torch.types import _size

from score_inverse.tasks.task import InverseTask


class ColorizationTask(InverseTask):
    def __init__(self, x_shape: _size):
        super().__init__()

        assert len(x_shape) == 3
        self.c, self.h, self.w = x_shape

        # We need the kernel to be orthogonal (up to some scaling constant),
        # such that the first column of `kernel_inv` is proportional to
        # [1, 1, 1]. This is desired since we will modify the grey channel
        # when sampling the inverse. Having the first column as [1, 1, 1] means
        # that the modifications are evenly distributed across the R, G, B channels.
        # We do so by choosing normal vectors for each row in `kernel`.

        v_gray = torch.tensor([1 / 3, 1 / 3, 1 / 3])  # Gray = (R + G + B) / 3
        v_1 = torch.tensor([1.0, -1.0, 0.0])  # Some vector normal to v_gray
        v_2 = torch.cross(v_gray, v_1)  # Some vector normal to v_gray and v_1

        self.kernel = torch.stack([v_gray, v_1, v_2])
        self.kernel_inv = torch.inverse(self.kernel)

    def noise(self, n: int) -> torch.Tensor:
        return torch.tensor(0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.view((3, 3, 1, 1))  # (out in kernel_h kernel_w)
        return F.conv2d(x, kernel.to(x.device))

    def transform_inv(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel_inv.view((3, 3, 1, 1))  # (out in kernel_h kernel_w)
        return F.conv2d(x, kernel.to(x.device))

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, 1:, :, :] = 0
        return x

    def mask_inv(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, :1, :, :] = 0
        return x

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        return x[:, :1, :, :]

    def drop_inv(self, y: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(
            (y.shape[0], self.c, self.h, self.w), device=y.device, dtype=y.dtype
        )
        x[:, :1, :, :] = y
        return x
