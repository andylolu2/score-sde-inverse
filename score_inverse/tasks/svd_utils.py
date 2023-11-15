import torch
import torch.nn.functional as F
from torch import nn


class MemEfficientSVD(nn.Module):
    def __init__(self, A_row: torch.Tensor, A_col: torch.Tensor, A_ch: torch.Tensor):
        """Implements memory-efficient SVD when A can be decomposed several componentst.

        Specifically, A can be decomposed into
        A = A_row ⊗ A_col ⊗ A_ch  ((w, w') ⊗ (h, h') ⊗ (c, c')) = (w h c, w' h' c')

        Args:
            A_row: (w, w')
            A_col: (h, h')
            A_ch: (c, c')
        """
        super().__init__()

        self.h_, self.h = A_row.shape
        self.w_, self.w = A_col.shape
        self.c_, self.c = A_ch.shape

        self.A_row: torch.Tensor
        self.A_col: torch.Tensor
        self.A_ch: torch.Tensor
        self.register_buffer("A_row", A_row)
        self.register_buffer("A_col", A_col)
        self.register_buffer("A_ch", A_ch)

        self.As = [self.A_row, self.A_col, self.A_ch]
        self.dim_order = [2, 3, 1]

        for i, A in enumerate(self.As):
            U, S, V = torch.svd(A, some=False)
            self.register_buffer(f"U{i}", U)
            self.register_buffer(f"S{i}", S)
            self.register_buffer(f"V{i}", V)

    @property
    def Us(self):
        return [getattr(self, f"U{i}") for i in range(3)]

    @property
    def Ss(self):
        return [getattr(self, f"S{i}") for i in range(3)]

    @property
    def Vs(self):
        return [getattr(self, f"V{i}") for i in range(3)]

    def U(self, x: torch.Tensor) -> torch.Tensor:
        """Compute U @ x."""

        for dim, U in zip(self.dim_order, self.Us):
            # Bring dim to the second last dimension
            x = x.transpose(dim, -2)
            # Multiply by U
            x = U @ x
            # Bring dim back to its original position
            x = x.transpose(dim, -2)

        return x

    def Ut(self, x: torch.Tensor) -> torch.Tensor:
        """Compute U.T @ x.

        Args:
            x: (b, c, h, w)
        """

        for dim, U in zip(reversed(self.dim_order), reversed(self.Us)):
            # Bring dim to the second last dimension
            x = x.transpose(dim, -2)
            # Multiply by U.T
            x = U.t() @ x
            # Bring dim back to its original position
            x = x.transpose(dim, -2)

        return x

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute V @ x.

        Args:
            x: (b, c, h, w)
        """

        for dim, V in zip(self.dim_order, self.Vs):
            # Bring dim to the second last dimension
            x = x.transpose(dim, -2)
            # Multiply by V
            x = V @ x
            # Bring dim back to its original position
            x = x.transpose(dim, -2)

        return x

    def Vt(self, x: torch.Tensor) -> torch.Tensor:
        """Compute V.T @ x.

        Args:
            x: (b, c, h, w)
        """

        for dim, V in zip(self.dim_order, self.Vs):
            # Bring dim to the second last dimension
            x = x.transpose(dim, -2)
            # Multiply by V.T
            x = V.t() @ x
            # Bring dim back to its original position
            x = x.transpose(dim, -2)

        return x

    def S(self, x: torch.Tensor) -> torch.Tensor:
        """Compute S @ x.

        Args:
            x: (b, c, h, w)
        """

        for dim, S in zip(self.dim_order, self.Ss):
            # Bring dim to the last dimension
            x = x.transpose(dim, -1)
            # Multiply by S
            x = S * x[..., : S.shape[0]]
            # Bring dim back to its original position
            x = x.transpose(dim, -1)

        return x

    def S_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Compute S^{-1} @ x.

        Args:
            x: (b, c, h, w)
        """

        for dim, S in zip(self.dim_order, self.Ss):
            # Bring dim to the last dimension
            x = x.transpose(dim, -1)
            # Multiply by S^{-1}
            x = x / S
            # Bring dim back to its original position
            x = x.transpose(dim, -1)

        x = F.pad(x, (0, self.w - self.w_, 0, self.h - self.h_, 0, self.c - self.c_))

        return x

    def A(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A @ x.

        Args:
            x: (b, c, h, w)
        """

        x = self.Vt(x)
        x = self.S(x)
        x = self.U(x)

        return x

    def A_pinv(self, x: torch.Tensor) -> torch.Tensor:
        """Compute A^{-1} @ x.

        Args:
            x: (b, c, h, w)
        """

        x = self.Ut(x)
        x = self.S_inv(x)
        x = self.V(x)

        return x
