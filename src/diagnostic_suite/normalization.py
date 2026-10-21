from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import PairDataset


@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> "Normalizer":
        return Normalizer(self.mean.to(device), self.std.to(device))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return (x - self.mean.view(*shape)) / self.std.view(*shape)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.std.view(*shape) + self.mean.view(*shape)


def estimate_normalizer(dataset: PairDataset, n_samples: int = 256) -> Normalizer:
    idxs = list(range(min(len(dataset), n_samples)))
    sums = None
    sums2 = None
    count = 0
    for i in idxs:
        item = dataset[i]
        for key in ("x", "y"):
            x = item[key].float()
            dims = tuple(range(1, x.ndim))
            s = x.sum(dim=dims)
            s2 = (x * x).sum(dim=dims)
            n = int(np.prod(x.shape[1:]))
            sums = s if sums is None else sums + s
            sums2 = s2 if sums2 is None else sums2 + s2
            count += n
    assert sums is not None and sums2 is not None
    mean = sums / max(1, count)
    var = sums2 / max(1, count) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return Normalizer(mean.float(), std.float())
