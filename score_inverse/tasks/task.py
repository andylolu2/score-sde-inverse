import abc

import torch


class InverseTask(abc.ABC):
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

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Implements `A @ x = P(Λ) @ T @ x` from the paper."""
        y = self.drop(self.transform(x))
        return y
