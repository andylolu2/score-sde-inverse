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
        return self.add_noise(self.A(x))

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the `ϵ` from the paper."""
        return x

    @property
    @abc.abstractmethod
    def output_shape(self):
        """Returns shape after applying `A`."""
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

    @property
    def output_shape(self):
        return (self.A_ch.shape[0], self.A_row.shape[0], self.A_col.shape[0])

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
    def __init__(self, tasks: list[DecomposeddSVDInverseTask]):
        InverseTask.__init__(self)
        nn.Module.__init__(self)

        self.x_shape = tasks[0].x_shape
        self.tasks = nn.ModuleList(tasks)
        self.svd = MemEfficientSVD(self.A_row, self.A_col, self.A_ch)

    @property
    def A_row(self):
        # Combine the row-wise operators of both tasks
        curr = torch.eye(self.x_shape[1])
        for task in self.tasks:
            curr = torch.matmul(task.A_row, curr)
        return curr

    @property
    def A_col(self):
        # Combine the column-wise operators of both tasks
        curr = torch.eye(self.x_shape[2])
        for task in self.tasks:
            curr = torch.matmul(task.A_col, curr)
        return curr

    @property
    def A_ch(self):
        # Combine the channel-wise operators of both tasks
        curr = torch.eye(self.x_shape[0])
        for task in self.tasks:
            curr = torch.matmul(task.A_ch, curr)
        return curr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `A @ x + ϵ` from the paper."""
        for task in self.tasks:
            x = task.forward(x)
        return x

    def add_noise(self, x: torch.Tensor):
        raise NotImplementedError()
