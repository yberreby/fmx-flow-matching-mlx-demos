import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mlx.core as mx
import tyro

matplotlib.use("Agg")

from fmx.checkpoint import save
from fmx.data import Toy, toy_sampler
from fmx.nets import MLPConfig
from fmx.plot import reveal
from fmx.plot.toy import dashboard
from fmx.train import progress_line, train


@dataclass
class Config:
    dataset: Toy = "two_moons"
    steps: int = 5000
    batch: int = 512
    lr: float = 3e-4
    width: int = 256
    depth: int = 4
    seed: int = 0
    out_dir: Path = Path("outputs")
    open_plot: bool = True


def main(cfg: Config) -> None:
    mx.random.seed(cfg.seed)
    mcfg = MLPConfig(width=cfg.width, depth=cfg.depth)
    model = mcfg.build()
    sampler = toy_sampler(cfg.dataset)

    t0 = time.time()
    history = train(
        model, sampler, steps=cfg.steps, batch=cfg.batch, lr=cfg.lr,
        on_log=lambda i, m: print(progress_line(i, m)),
    )
    print(f"trained {cfg.steps} steps in {time.time() - t0:.1f}s")

    run_dir = cfg.out_dir / f"toy2d_{cfg.dataset}"
    save(run_dir / "ckpt", model, mcfg,
         meta={"dataset": cfg.dataset, "steps": cfg.steps, "final_loss": history[-1][1]})
    plot = dashboard(model, sampler(5000), history, run_dir / "dashboard.png")
    reveal(plot, open_it=cfg.open_plot)


if __name__ == "__main__":
    main(tyro.cli(Config))
