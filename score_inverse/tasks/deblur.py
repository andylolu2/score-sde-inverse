from typing import Callable, Literal
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn
from torch.types import _size
from scipy import signal

from score_inverse.tasks.task import InverseTask, DecomposeddSVDInverseTask
from .svd_utils import MemEfficientSVD


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
                j_pos = i - kernel.shape[0] // 2 + j
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


# def flatten_input_output(
#     f: Callable, x: torch.Tensor, output_shape: _size | None = None
# ):
#     shape = x.shape[-2:] if output_shape is None else output_shape
#     x = x.flatten(-2)
#     out = f(x)
#     return out.unflatten(-1, shape)


# def blur_kernel(kernel_type: str, kernel_size: int) -> torch.Tensor:
#     if kernel_type == "gaussian":
#         std = 10
#         kernel_1d = signal.windows.gaussian(kernel_size, std=std)
#         kernel_1d = torch.from_numpy(kernel_1d)
#     elif kernel_type == "uniform":
#         kernel_1d = torch.ones(kernel_size)
#     else:
#         raise ValueError(f"Invalid kernel type: {kernel_type}")

#     return kernel_1d / torch.sum(kernel_1d)

#     kernel = torch.outer(kernel_1d, kernel_1d)
#     kernel = kernel / torch.sum(kernel)
#     return kernel


# def linearize_kernel(
#     kernel: torch.Tensor, img_dim: int, blurred_dim: int
# ) -> torch.Tensor:
#     # kh, kw = kernel.shape
#     A_small = torch.zeros(img_dim, blurred_dim)
#     for i in range(img_dim):
#         for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
#             if j < 0 or j >= img_dim:
#                 continue
#             A_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
#     return A_small

#     # A = torch.zeros(*img_size, *blurred_size)
#     # for i, j in product(range(img_size[0]), range(img_size[1])):
#     #     for m, n in product(range(kh), range(kw)):
#     #         dst_m = i - 2 * (kh // 2) + m
#     #         dst_n = j - 2 * (kw // 2) + n

#     #         if dst_m < 0 or dst_m >= A.shape[2] or dst_n < 0 or dst_n >= A.shape[3]:
#     #             continue

#     #         A[i, j, dst_m, dst_n] = kernel[m, n]

#     # A = A.reshape(A.shape[0] * A.shape[1], A.shape[2] * A.shape[3])
#     # return A


# class DeblurTask(InverseTask, nn.Module):
#     """This implementation of the deblurring diverges from the framework proposed by the original paper.

#     We make use SVD to find a decomposition of A = T @ P(Λ) = T' @ Σ @ V.
#     Note: We are using A as a left-operand here for ease of implementation.

#     We can substitute T ~= T' and P(Λ) ~= Σ @ V. Turns out all the math work out
#     exactly the same. However, this breaks the assumption that P(Λ) only "drops"
#     some features while keeping others the same. But hey, it works ¯\_(ツ)_/¯.
#     """

#     kernel: torch.Tensor
#     U_small: torch.Tensor
#     S: torch.Tensor
#     V_small: torch.Tensor
#     U_small_inv: torch.Tensor
#     S_inv: torch.Tensor
#     V_small_inv: torch.Tensor
#     perm: torch.Tensor

#     def __init__(
#         self,
#         x_shape: _size,
#         kernel_size: int = 5,
#         kernel_type: Literal["uniform", "gaussian"] = "gaussian",
#     ):
#         InverseTask.__init__(self)
#         nn.Module.__init__(self)

#         assert len(x_shape) == 3
#         self.c, self.h, self.w = x_shape
#         self.kernel_size = kernel_size
#         self.kernel_type = kernel_type

#         assert self.h == self.w

#         self.img_dim = self.h
#         self.blurred_dim = self.img_dim - 2 * (kernel_size // 2)

#         self.register_buffer("kernel", blur_kernel(kernel_type, kernel_size))

#         A_small = linearize_kernel(self.kernel, self.img_dim, self.blurred_dim)
#         U_small, S_small, V_small = torch.svd(A_small, some=False)
#         S = torch.outer(S_small, S_small).reshape(self.img_dim**2)
#         S, perm = self.S.sort(descending=True)

#         self.register_buffer("U_small", U_small)
#         self.register_buffer("S", S)
#         self.register_buffer("V_small", V_small)
#         self.register_buffer("U_small_inv", U_small.t())
#         self.register_buffer("V_small_inv", V_small.t())
#         self.register_buffer("perm", perm)

#         # T, S, V = torch.linalg.svd(A)
#         # S = torch.diag(S)
#         # self.register_buffer("T", T)
#         # self.register_buffer("S", S)
#         # self.register_buffer("V", V)
#         # self.register_buffer("T_inv", self.T.t())
#         # self.register_buffer("S_inv", torch.linalg.inv(self.S))
#         # self.register_buffer("V_inv", self.V.t())

#     def noise(self, n: int) -> torch.Tensor:
#         return torch.tensor(0)

#     def transform(self, x: torch.Tensor) -> torch.Tensor:
#         # invert the permutation
#         temp = torch.zeros(x.shape[0], self.c, self.img_dim**2, device=x.device)
#         temp[:, :, self.perm] = x.reshape(x.shape[0], self.c, self.img_dim**2)
#         temp = temp.permute(0, 2, 1)
#         # multiply the image by U from the left and by U^T from the right
#         out = self.mat_by_img(self.U_small, temp)
#         out = self.img_by_mat(out, self.U_small_inv).reshape(x.shape[0], -1)
#         return out
#         return flatten_input_output(lambda flat: flat @ self.T, x)

#     def transform_inv(self, x: torch.Tensor) -> torch.Tensor:
#         return flatten_input_output(lambda flat: flat @ self.T_inv, x)

#     def mask(self, x: torch.Tensor) -> torch.Tensor:
#         def f(x: torch.Tensor):
#             x = x.clone()
#             x[..., self.blurred_size.numel() :] = 0
#             return x

#         return flatten_input_output(f, x)

#     def mask_inv(self, x: torch.Tensor) -> torch.Tensor:
#         def f(x: torch.Tensor):
#             x = x.clone()
#             x[..., : self.blurred_size.numel()] = 0
#             return x

#         return flatten_input_output(f, x)

#     def drop(self, x: torch.Tensor) -> torch.Tensor:
#         return flatten_input_output(
#             lambda flat: flat[..., : self.blurred_size.numel()] @ self.S @ self.V,
#             x,
#             self.blurred_size,
#         )

#     def drop_inv(self, y: torch.Tensor) -> torch.Tensor:
#         return flatten_input_output(
#             lambda flat: F.pad(
#                 flat @ self.V_inv @ self.S_inv,
#                 (0, self.img_size.numel() - self.blurred_size.numel()),
#             ),
#             y,
#             self.img_size,
#         )

#     def mat_by_img(self, M: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
#         return torch.matmul(
#             M, img.reshape(img.shape[0] * self.c, self.img_dim, self.img_dim)
#         ).reshape(img.shape[0], self.c, M.shape[0], self.img_dim)

#     def img_by_mat(self, img: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
#         return torch.matmul(
#             img.reshape(img.shape[0] * self.c, self.img_dim, self.img_dim), M
#         ).reshape(img.shape[0], self.c, self.img_dim, M.shape[1])
