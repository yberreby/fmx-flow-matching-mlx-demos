from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.core import array

from ..data.mnist import MNIST_SHAPE
from ..sample import sample
from . import panel_loss, save_fig


def _to_img(x: array) -> np.ndarray:
    # (..., H, W, 1) in [-1, 1] -> (..., H, W) in [0, 1]
    return np.clip((np.asarray(x[..., 0]) + 1.0) * 0.5, 0.0, 1.0)


def sample_grid(model: nn.Module, k: int, n_steps: int = 50, seed: int = 0) -> array:
    mx.random.seed(seed)
    return sample(model, mx.random.normal((k * k, *MNIST_SHAPE)), n_steps=n_steps)[-1]


def grid_canvas(samples: array, k: int) -> np.ndarray:
    imgs = _to_img(samples)  # (k*k, H, W)
    _, h, w = imgs.shape
    return imgs.reshape(k, k, h, w).transpose(0, 2, 1, 3).reshape(k * h, k * w)


def panel_grid(ax, samples: array, k: int) -> None:
    ax.imshow(grid_canvas(samples, k), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{k}x{k} samples")
    ax.axis("off")


def panel_trajectory(ax, model: nn.Module, n_rows: int = 6, n_cols: int = 8,
                     n_steps: int = 50, seed: int = 1) -> None:
    mx.random.seed(seed)
    traj = sample(model, mx.random.normal((n_rows, *MNIST_SHAPE)), n_steps=n_steps)
    picks = mx.array(np.linspace(0, traj.shape[0] - 1, n_cols).round().astype(int))
    strip = _to_img(traj[picks])  # (n_cols, n_rows, H, W)
    _, _, h, w = strip.shape
    canvas = strip.transpose(1, 2, 0, 3).reshape(n_rows * h, n_cols * w)
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    ax.set_title("noise -> digit (left to right)")
    ax.axis("off")


def dashboard(model: nn.Module, history, out: Path, k: int = 8,
              n_steps: int = 50, seed: int = 0) -> Path:
    fig, (ax_grid, ax_traj, ax_loss) = plt.subplots(1, 3, figsize=(20, 7))
    panel_grid(ax_grid, sample_grid(model, k, n_steps=n_steps, seed=seed), k)
    panel_trajectory(ax_traj, model, n_steps=n_steps)
    panel_loss(ax_loss, history)
    fig.tight_layout()
    return save_fig(fig, out)


def save_trajectory_strip(model: nn.Module, out: Path, n_rows: int = 8,
                          n_cols: int = 10, n_steps: int = 50, seed: int = 1) -> Path:
    fig, ax = plt.subplots(figsize=(n_cols, n_rows))
    panel_trajectory(ax, model, n_rows=n_rows, n_cols=n_cols, n_steps=n_steps, seed=seed)
    fig.tight_layout()
    return save_fig(fig, out)
