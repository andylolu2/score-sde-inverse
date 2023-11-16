import abc
from typing import Union, List, Tuple

import torch
from torch import nn, Size
from torch.types import _size

from .svd_utils import MemEfficientSVD


class InverseTask(abc.ABC):
    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `A @ x = P(Λ) @ T @ x` from the paper."""
        y = self.drop(self.transform(x))
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `A @ x + ϵ` from the paper."""
        return self.A(x) + self.noise(x.shape[0])

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

    @abc.abstractmethod
    def get_output_shape(self):
        """Returns shape after applying `A`."""
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

    def get_output_shape(self):
        return self.x_shape


class CombinedTask(DecomposeddSVDInverseTask):

    def __init__(self, tasks: list[DecomposeddSVDInverseTask]):
        assert len(tasks) >= 2, "The list of tasks should contain at least two tasks"
        # Call the initializers of the base classes first
        InverseTask.__init__(self)
        nn.Module.__init__(self)

        # Assign the tasks
        self.tasks = tasks

        # Set the x_shape based on the task shapes
        self.x_shape = tasks[0].x_shape

        # Initialize the combined SVD using the composed operators
        self.svd = MemEfficientSVD(self.A_row, self.A_col, self.A_ch)

    @property
    def A_row(self):
        # Combine the row-wise operators of both tasks
        curr = self.tasks[0].A_row
        for task in self.tasks[1:]:
            curr = torch.matmul(task.A_row, curr)
        return curr

    @property
    def A_col(self):
        # Combine the column-wise operators of both tasks
        curr = self.tasks[0].A_col
        for task in self.tasks[1:]:
            curr = torch.matmul(task.A_col, curr)
        return curr

    @property
    def A_ch(self):
        # Combine the channel-wise operators of both tasks
        curr = self.tasks[0].A_ch
        for task in self.tasks[1:]:
            curr = torch.matmul(task.A_ch, curr)
        return curr

    def noise(self, n):
        """A_2 ( A_1 x + noise_1) + noise_2

        A_2 A_1 x + (A_2 noise_1 + noise_2)
        """
        combined_noise = self.tasks[0].noise(n)
        for task in self.tasks[1:]:
            combined_noise = task.A(combined_noise) + task.noise(n)

        return combined_noise



