from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array

Solver = Literal["euler", "heun"]


def sample(model: nn.Module, x0: array, *, n_steps: int,
           solver: Solver = "euler") -> array:
    # integrate dx/dt = v(t, x) from t=0 (noise) to t=1 (data); return the whole
    # trajectory (n_steps + 1, *x0.shape). A single eval at the end lets MLX fuse
    # the unrolled ODE instead of syncing every step.
    dt = 1.0 / n_steps
    x = x0
    traj = [x]
    for i in range(n_steps):
        t = mx.full((x.shape[0],), i * dt)
        v = model(t, x)
        if solver == "heun":
            t_next = mx.full((x.shape[0],), (i + 1) * dt)
            v = 0.5 * (v + model(t_next, x + dt * v))
        x = x + dt * v
        traj.append(x)
    out = mx.stack(traj, axis=0)
    mx.eval(out)
    return out
