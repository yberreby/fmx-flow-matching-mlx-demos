import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import tyro

from fmx.coupling import couple_ot, straightness
from fmx.data import Toy, toy_sampler
from fmx.nets import MLPConfig
from fmx.plot import reveal, save_fig
from fmx.plot.toy import panel_samples
from fmx.sample import sample
from fmx.train import Couple, train


@dataclass
class Config:
    dataset: Toy = "eight_gaussians"
    steps: int = 6000
    batch: int = 256
    seed: int = 0
    out: Path = Path("outputs/coupling.png")
    open_plot: bool = True


def _trained_model(cfg: Config, couple: Couple | None):
    mx.random.seed(cfg.seed)
    model = MLPConfig().build()
    t0 = time.time()
    train(model, toy_sampler(cfg.dataset), steps=cfg.steps, batch=cfg.batch,
          couple=couple, log_every=cfg.steps)
    return model, time.time() - t0


def main(cfg: Config) -> None:
    data = toy_sampler(cfg.dataset)(5000)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (label, couple) in enumerate([("independent", None), ("OT", couple_ot)]):
        model, dt = _trained_model(cfg, couple)
        print(f"{label}: trained in {dt:.1f}s")
        traj = sample(model, mx.random.normal((2000, 2)), n_steps=50)
        s = np.asarray(straightness(traj))
        panel_samples(axes[row, 0], traj[-1], data)
        axes[row, 0].set_title(f"{label}: 50-step samples")
        axes[row, 1].hist(s, bins=40, range=(0.4, 1.0), color="steelblue")
        axes[row, 1].set_title(f"{label}: straightness (median {np.median(s):.3f})")
        axes[row, 1].set_xlabel("displacement / path-length")
        # 2-step generation is where straight trajectories pay off.
        gen2 = sample(model, mx.random.normal((2000, 2)), n_steps=2)[-1]
        panel_samples(axes[row, 2], gen2, data)
        axes[row, 2].set_title(f"{label}: 2-step samples")
    fig.tight_layout()
    reveal(save_fig(fig, cfg.out), open_it=cfg.open_plot)


if __name__ == "__main__":
    main(tyro.cli(Config))
