from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array

from .embed import fourier_time_embed


class VelocityMLP(nn.Module):
    def __init__(self, cfg: "MLPConfig"):
        super().__init__()
        self.n_time_freqs = cfg.n_time_freqs
        sizes = [cfg.dim + 2 * cfg.n_time_freqs] + [cfg.width] * cfg.depth
        self.layers = [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:])]
        self.out = nn.Linear(cfg.width, cfg.dim)

    def __call__(self, t: array, x: array) -> array:
        h = mx.concatenate([x, fourier_time_embed(t, self.n_time_freqs)], axis=-1)
        for layer in self.layers:
            h = nn.silu(layer(h))
        return self.out(h)


@dataclass(frozen=True)
class MLPConfig:
    dim: int = 2
    width: int = 256
    depth: int = 4
    n_time_freqs: int = 16

    def build(self) -> VelocityMLP:
        return VelocityMLP(self)
