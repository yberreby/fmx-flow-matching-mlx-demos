import math
from typing import Callable, Literal

import mlx.core as mx
from mlx.core import array

Toy = Literal["two_moons", "eight_gaussians", "spiral", "checkerboard"]


def two_moons(n: int, noise: float = 0.1) -> array:
    half = n // 2
    t_out = mx.random.uniform(0, math.pi, (half,))
    outer = mx.stack([mx.cos(t_out), mx.sin(t_out)], axis=-1)
    t_in = mx.random.uniform(0, math.pi, (n - half,))
    inner = mx.stack([1.0 - mx.cos(t_in), 0.5 - mx.sin(t_in)], axis=-1)
    x = mx.concatenate([outer, inner], axis=0) + noise * mx.random.normal((n, 2))
    return (x - mx.array([0.5, 0.25])) * 1.5  # center, scale to ~unit spread


def eight_gaussians(n: int, std: float = 0.1, radius: float = 2.0) -> array:
    angles = mx.arange(8) * (2 * math.pi / 8)
    centers = radius * mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)
    return centers[mx.random.randint(0, 8, (n,))] + std * mx.random.normal((n, 2))


def spiral(n: int, noise: float = 0.05, turns: float = 1.5) -> array:
    r = mx.random.uniform(0, 1, (n,)) ** 0.5
    angle = turns * 2 * math.pi * r
    x = mx.stack([r * mx.cos(angle), r * mx.sin(angle)], axis=-1)
    return (x + noise * mx.random.normal((n, 2))) * 3.0


def checkerboard(n: int) -> array:
    # 4x4 board on [-2, 2]^2; pick one of the 8 "on" cells (i + j even), then
    # uniform within it. Constructive, so no points leak into empty cells.
    on = mx.array([[i, j] for i in range(4) for j in range(4) if (i + j) % 2 == 0])
    cell = on[mx.random.randint(0, on.shape[0], (n,))]
    return -2.0 + cell + mx.random.uniform(0, 1, (n, 2))


_REGISTRY: dict[Toy, Callable[[int], array]] = {
    "two_moons": two_moons,
    "eight_gaussians": eight_gaussians,
    "spiral": spiral,
    "checkerboard": checkerboard,
}


def toy_sampler(name: Toy) -> Callable[[int], array]:
    return _REGISTRY[name]
