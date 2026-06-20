import mlx.core as mx
import numpy as np
from mlx.core import array
from scipy.optimize import linear_sum_assignment


def couple_ot(x0: array, x1: array) -> tuple[array, array]:
    # minibatch optimal transport: reorder x1 to the minimum-squared-distance
    # assignment against x0. Pairing near points straightens the trajectories
    # the model has to learn. 2D points only.
    assert x0.ndim == 2 and x1.ndim == 2
    a, b = np.asarray(x0), np.asarray(x1)
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    row, col = linear_sum_assignment(cost)
    return x0[mx.array(row)], x1[mx.array(col)]


def straightness(traj: array) -> array:
    # per-trajectory displacement / path-length in (0, 1]; 1 is a straight line.
    # traj: (n_steps + 1, N, d).
    seg = mx.sqrt(((traj[1:] - traj[:-1]) ** 2).sum(-1)).sum(0)
    disp = mx.sqrt(((traj[-1] - traj[0]) ** 2).sum(-1))
    return disp / (seg + 1e-9)
