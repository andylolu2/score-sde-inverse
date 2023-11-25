import torch

from score_inverse.tasks.task import DecomposeddSVDInverseTask


class ColorizationTask(DecomposeddSVDInverseTask):
    @property
    def A_ch(self) -> torch.Tensor:
        return torch.tensor([[0.3333, 0.3333, 0.3334]])
