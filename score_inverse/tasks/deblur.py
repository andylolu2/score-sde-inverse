from typing import Callable, Literal
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn
from torch.types import _size
from scipy import signal

from score_inverse.tasks.task import InverseTask


def flatten_input_output(
    f: Callable, x: torch.Tensor, output_shape: _size | None = None
):
    shape = x.shape[-2:] if output_shape is None else output_shape
    x = x.flatten(-2)
    out = f(x)
    return out.unflatten(-1, shape)


def blur_kernel(kernel_type: str, kernel_size: int) -> torch.Tensor:
    if kernel_type == "gaussian":
        std = 1
        kernel_1d = signal.windows.gaussian(kernel_size, std=std)
        kernel_1d = torch.from_numpy(kernel_1d)
    elif kernel_type == "uniform":
        kernel_1d = torch.ones(kernel_size)
    else:
        raise ValueError(f"Invalid kernel type: {kernel_type}")

    kernel = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel / torch.sum(kernel)
    return kernel


def linearize_kernel(
    kernel: torch.Tensor, img_size: torch.Size, blurred_size: torch.Size
) -> torch.Tensor:
    kh, kw = kernel.shape
    A = torch.zeros(*img_size, *blurred_size)
    for i, j in product(range(img_size[0]), range(img_size[1])):
        for m, n in product(range(kh), range(kw)):
            dst_m = i - 2 * (kh // 2) + m
            dst_n = j - 2 * (kw // 2) + n

            if dst_m < 0 or dst_m >= A.shape[2] or dst_n < 0 or dst_n >= A.shape[3]:
                continue

            A[i, j, dst_m, dst_n] = kernel[m, n]

    A = A.reshape(A.shape[0] * A.shape[1], A.shape[2] * A.shape[3])
    return A


class DeblurTask(InverseTask, nn.Module):
    """This implementation of the deblurring diverges from the framework proposed by the original paper.

    We make use SVD to find a decomposition of A = T @ P(Λ) = T' @ Σ @ V.
    Note: We are using A as a left-operand here for ease of implementation.

    We can substitute T ~= T' and P(Λ) ~= Σ @ V. Turns out all the math work out
    exactly the same. However, this breaks the assumption that P(Λ) only "drops"
    some features while keeping others the same. But hey, it works ¯\_(ツ)_/¯.
    """

    kernel: torch.Tensor
    T: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor
    T_inv: torch.Tensor
    S_inv: torch.Tensor
    V_inv: torch.Tensor

    def __init__(
        self,
        x_shape: _size,
        kernel_size: int = 5,
        kernel_type: Literal["uniform", "gaussian"] = "gaussian",
    ):
        InverseTask.__init__(self)
        nn.Module.__init__(self)

        assert len(x_shape) == 3
        self.c, self.h, self.w = x_shape
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type

        self.img_size = torch.Size((self.h, self.w))
        self.blurred_size = torch.Size(
            (
                self.h - 2 * (kernel_size // 2),
                self.w - 2 * (kernel_size // 2),
            )
        )

        self.register_buffer("kernel", blur_kernel(kernel_type, kernel_size))

        A = linearize_kernel(self.kernel, self.img_size, self.blurred_size)

        T, S, V = torch.linalg.svd(A)
        S = torch.diag(S)
        self.register_buffer("T", T)
        self.register_buffer("S", S)
        self.register_buffer("V", V)
        self.register_buffer("T_inv", self.T.t())
        self.register_buffer("S_inv", torch.linalg.inv(self.S))
        self.register_buffer("V_inv", self.V.t())

    def noise(self, n: int) -> torch.Tensor:
        return torch.tensor(0)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return flatten_input_output(lambda flat: flat @ self.T, x)

    def transform_inv(self, x: torch.Tensor) -> torch.Tensor:
        return flatten_input_output(lambda flat: flat @ self.T_inv, x)

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        def f(x: torch.Tensor):
            x = x.clone()
            x[..., self.blurred_size.numel() :] = 0
            return x

        return flatten_input_output(f, x)

    def mask_inv(self, x: torch.Tensor) -> torch.Tensor:
        def f(x: torch.Tensor):
            x = x.clone()
            x[..., : self.blurred_size.numel()] = 0

        return flatten_input_output(f, x)

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        return flatten_input_output(
            lambda flat: flat[..., : self.blurred_size.numel()] @ self.S @ self.V,
            x,
            self.blurred_size,
        )

    def drop_inv(self, y: torch.Tensor) -> torch.Tensor:
        return flatten_input_output(
            lambda flat: F.pad(
                flat @ self.V_inv @ self.S_inv,
                (0, self.img_size.numel() - self.blurred_size.numel()),
            ),
            y,
            self.img_size,
        )
