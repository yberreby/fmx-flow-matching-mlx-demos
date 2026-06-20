import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mlx.core as mx
import tyro

matplotlib.use("Agg")

from fmx.checkpoint import save
from fmx.data import batch_sampler, load_mnist
from fmx.nets import UNetConfig
from fmx.plot import reveal
from fmx.plot.image import dashboard
from fmx.train import progress_line, train


@dataclass
class Config:
    steps: int = 6000
    batch: int = 256
    lr: float = 5e-4
    base: int = 32
    n_steps_sample: int = 50
    save_every: int = 500
    seed: int = 0
    out_dir: Path = Path("outputs")
    open_plot: bool = True


def main(cfg: Config) -> None:
    mx.random.seed(cfg.seed)
    sampler = batch_sampler(load_mnist())
    mcfg = UNetConfig(base=cfg.base)
    model = mcfg.build()
    run_dir = cfg.out_dir / "mnist"

    def on_log(i: int, m: dict) -> None:
        print(progress_line(i, m))
        # overwrite a single canonical ckpt so time_slider can load it mid-train.
        save(run_dir / "ckpt", model, mcfg, meta={"steps": i, "loss": m["loss"]})

    t0 = time.time()
    history = train(model, sampler, steps=cfg.steps, batch=cfg.batch, lr=cfg.lr,
                    log_every=cfg.save_every, on_log=on_log)
    dt = time.time() - t0
    print(f"trained {cfg.steps} steps in {dt:.1f}s ({dt / cfg.steps * 1e3:.1f} ms/step)")

    save(run_dir / "ckpt", model, mcfg,
         meta={"steps": cfg.steps, "final_loss": history[-1][1]})
    plot = dashboard(model, history, run_dir / "dashboard.png",
                     n_steps=cfg.n_steps_sample, seed=cfg.seed)
    reveal(plot, open_it=cfg.open_plot)


if __name__ == "__main__":
    main(tyro.cli(Config))
