from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import mlx.nn as nn
from mlx.core import array

from ..sample import sample
from . import panel_loss, save_fig


def panel_field(ax, model: nn.Module, t: float, lim: float = 3.0, n: int = 24) -> None:
    g = mx.linspace(-lim, lim, n)
    gx, gy = mx.meshgrid(g, g)
    pts = mx.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    v = np.asarray(model(mx.full((pts.shape[0],), t), pts))
    u, w = v[:, 0].reshape(n, n), v[:, 1].reshape(n, n)
    ax.quiver(np.asarray(gx), np.asarray(gy), u, w, np.hypot(u, w),
              cmap="viridis", scale=40)
    ax.set_title(f"v(t={t:.2f})")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")


def panel_samples(ax, gen: array, data: array, lim: float = 3.5) -> None:
    d = np.asarray(data)
    ax.hist2d(d[:, 0], d[:, 1], bins=80, range=[[-lim, lim], [-lim, lim]],
              cmap="Greys")
    g = np.asarray(gen)
    ax.scatter(g[:, 0], g[:, 1], s=2, c="crimson", alpha=0.4)
    ax.set_title("generated (red) over target")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")


def panel_trajectories(ax, traj: array, lim: float = 3.5, k: int = 200) -> None:
    tr = np.asarray(traj[:, :k])  # (n_steps + 1, k, 2)
    steps = tr.shape[0]
    for s in range(steps - 1):
        ax.plot(tr[s:s + 2, :, 0], tr[s:s + 2, :, 1],
                color=plt.cm.plasma(s / max(steps - 1, 1)), lw=0.4, alpha=0.5)
    ax.scatter(tr[0, :, 0], tr[0, :, 1], s=3, c="blue", label="x0 (noise)")
    ax.scatter(tr[-1, :, 0], tr[-1, :, 1], s=3, c="red", label="x1 (data)")
    ax.set_title("trajectories")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)


def panel_step_sweep(ax, model: nn.Module, lim: float = 3.5,
                     steps_list: tuple[int, ...] = (1, 2, 4, 8, 64),
                     n_samples: int = 1500) -> None:
    for ns, col in zip(steps_list, plt.cm.cool(np.linspace(0, 1, len(steps_list)))):
        gen = np.asarray(sample(model, mx.random.normal((n_samples, 2)), n_steps=ns)[-1])
        ax.scatter(gen[:, 0], gen[:, 1], s=1, color=col, alpha=0.3, label=str(ns))
    ax.set_title("step-count sweep")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, title="steps")


def dashboard(model: nn.Module, data: array, history, out: Path,
              n_steps: int = 50) -> Path:
    traj = sample(model, mx.random.normal((2000, 2)), n_steps=n_steps)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    panel_loss(axes[0, 0], history)
    panel_samples(axes[0, 1], traj[-1], data)
    panel_trajectories(axes[0, 2], traj)
    panel_step_sweep(axes[0, 3], model)
    for ax, t in zip(axes[1], (0.0, 0.25, 0.5, 0.75)):
        panel_field(ax, model, t)
    fig.tight_layout()
    return save_fig(fig, out)
