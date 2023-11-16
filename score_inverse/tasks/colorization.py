import torch

from score_inverse.tasks.task import DecomposeddSVDInverseTask


class ColorizationTask(DecomposeddSVDInverseTask):
    def noise(self, n: int) -> torch.Tensor:
        return torch.zeros(n, *self.get_output_shape())

    def get_output_shape(self):
        return 1, self.x_shape[1], self.x_shape[2]

    @property
    def A_ch(self) -> torch.Tensor:
        return torch.tensor([[0.3333, 0.3333, 0.3334]])
