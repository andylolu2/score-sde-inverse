import abc

import torch
from torch import nn
from torch.types import _size

from .svd_utils import MemEfficientSVD


class InverseTask(abc.ABC):
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `A @ x = P(Λ) @ T @ x` from the paper."""
        y = self.drop(self.transform(x))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `A @ x + ϵ` from the paper."""
        return self.A(x) + self.noise(len(x))

    @abc.abstractmethod
    def noise(self, n: int) -> torch.Tensor:
        """Implements `ϵ` from the paper."""
        ...

    @abc.abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `T @ x` from the paper."""
        ...

    @abc.abstractmethod
    def transform_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `T^{-1} @ x` from the paper."""
        ...

    @abc.abstractmethod
    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `Λ @ x` from the paper."""
        ...

    @abc.abstractmethod
    def mask_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `(I - Λ) @ x` from the paper."""
        ...

    @abc.abstractmethod
    def drop(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `P(Λ) @ x` from the paper.

        Can return output of different shape.
        """
        ...

    @abc.abstractmethod
    def drop_inv(self, y: torch.Tensor) -> torch.Tensor:
        """Implements `P^{-1}(Λ) @ y` from the paper."""
        ...


class DecomposeddSVDInverseTask(InverseTask, nn.Module):
    def __init__(self, x_shape: _size):
        InverseTask.__init__(self)
        nn.Module.__init__(self)

        self.x_shape = x_shape

        self.svd = MemEfficientSVD(self.A_row, self.A_col, self.A_ch)

    @property
    def A_row(self) -> torch.Tensor:
        """The linear operator on the rows of the image."""
        return torch.eye(self.x_shape[1])

    @property
    def A_col(self) -> torch.Tensor:
        """The linear operator on the columns of the image."""
        return torch.eye(self.x_shape[2])

    @property
    def A_ch(self) -> torch.Tensor:
        """The linear operator on the channels of the image."""
        return torch.eye(self.x_shape[0])

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.svd.Vt(x)

    def transform_inv(self, x: torch.Tensor) -> torch.Tensor:
        return self.svd.V(x)

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        for dim, S in zip(self.svd.dim_order, self.svd.Ss):
            indices = torch.arange(S.shape[0], x.shape[dim], device=x.device)
            mask.index_fill_(dim, indices, True)
        x[mask] = 0
        return x

    def mask_inv(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        mask = torch.zeros_like(x, dtype=torch.bool)
        for dim, S in zip(self.svd.dim_order, self.svd.Ss):
            indices = torch.arange(S.shape[0], x.shape[dim], device=x.device)
            mask.index_fill_(dim, indices, True)
        x[~mask] = 0
        return x

    def drop(self, x: torch.Tensor) -> torch.Tensor:
        x = self.svd.S(x)
        x = self.svd.U(x)
        return x

    def drop_inv(self, y: torch.Tensor) -> torch.Tensor:
        x = self.svd.Ut(y)
        x = self.svd.S_inv(x)
        return x


class CombinedTask(DecomposeddSVDInverseTask):
    def __init__(self, task1, task2):
        # Call the initializers of the base classes first
        InverseTask.__init__(self)
        nn.Module.__init__(self)

        # Assign the tasks
        self.task1 = task1
        self.task2 = task2

        # Set the x_shape based on the task shapes
        self.x_shape = task1.x_shape

        # Initialize the combined SVD using the composed operators
        self.svd = MemEfficientSVD(self.A_row, self.A_col, self.A_ch)

    @property
    def A_row(self):
        # Combine the row-wise operators of both tasks
        return torch.matmul(self.task2.A_row, self.task1.A_row)

    @property
    def A_col(self):
        # Combine the column-wise operators of both tasks
        return torch.matmul(self.task2.A_col, self.task1.A_col)

    @property
    def A_ch(self):
        # Combine the channel-wise operators of both tasks
        return torch.matmul(self.task2.A_ch, self.task1.A_ch)

    def noise(self, n):
        """A_2 ( A_1 x + noise_1) + noise_2

        A_2 A_1 x + (A_2 noise_1 + noise_2)
        """
        return self.task2.A(self.task1.noise(n)) + self.task2.noise(n)
