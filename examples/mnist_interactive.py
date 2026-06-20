import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.optimizers as optim
import tyro

from fmx.checkpoint import save
from fmx.data import batch_sampler, load_mnist
from fmx.nets import UNetConfig
from fmx.plot.image import grid_canvas, sample_grid
from fmx.train import make_step

LR_FACTOR = 2.0
SAMPLE_STEP_BOUNDS = (1, 200)


@dataclass
class Config:
    batch: int = 256
    lr: float = 5e-4
    base: int = 32
    k: int = 6
    sample_steps: int = 20
    redraw_every: int = 50
    seed: int = 0
    out: Path = Path("outputs/mnist/ckpt")


class Trainer:
    # one MNIST UNet you can train, pause, reset, and retune live. A fresh Adam
    # is built whenever the LR changes because mx.compile captures optimizer
    # state, so mutating lr in place would not take effect.
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sampler = batch_sampler(load_mnist())
        self.mcfg = UNetConfig(base=cfg.base)
        self.lr = cfg.lr
        self.sample_steps = cfg.sample_steps
        self.paused = False
        self.quit = False
        self.history: list[tuple[int, float]] = []
        self.reset()

    def reset(self) -> None:
        mx.random.seed(self.cfg.seed)
        self.model = self.mcfg.build()
        self.step = 0
        self.history.clear()
        self._rebuild_optimizer()
        print("[reset]", flush=True)

    def _rebuild_optimizer(self) -> None:
        self.opt = optim.Adam(learning_rate=self.lr)
        self.step_fn = make_step(self.model, self.opt)

    def train_step(self) -> tuple[float, float]:
        x1 = self.sampler(self.cfg.batch)
        x0 = mx.random.normal(x1.shape)
        t = mx.random.uniform(0, 1, (self.cfg.batch,))
        loss, grad_norm = self.step_fn(x0, x1, t)
        self.step += 1
        return loss.item(), grad_norm.item()

    def grid(self):
        samples = sample_grid(self.model, self.cfg.k, n_steps=self.sample_steps,
                              seed=self.cfg.seed)
        return grid_canvas(samples, self.cfg.k)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        print(f"[{'paused' if self.paused else 'resumed'}]", flush=True)

    def scale_lr(self, factor: float) -> None:
        self.lr *= factor
        self._rebuild_optimizer()
        print(f"lr -> {self.lr:.2e}", flush=True)

    def scale_sample_steps(self, factor: float) -> None:
        lo, hi = SAMPLE_STEP_BOUNDS
        self.sample_steps = int(min(max(self.sample_steps * factor, lo), hi))
        print(f"sample_steps -> {self.sample_steps}", flush=True)

    def keymap(self) -> dict[str, Callable[[], None]]:
        return {
            " ": self.toggle_pause,
            "r": self.reset,
            "]": lambda: self.scale_lr(LR_FACTOR),
            "[": lambda: self.scale_lr(1 / LR_FACTOR),
            "+": lambda: self.scale_sample_steps(2),
            "=": lambda: self.scale_sample_steps(2),
            "-": lambda: self.scale_sample_steps(0.5),
        }


def main(cfg: Config) -> None:
    import matplotlib
    matplotlib.use("macosx", force=True)
    import matplotlib.pyplot as plt

    print(f"MLX device: {mx.default_device()}", flush=True)
    print("keys: q=quit  space=pause  r=reset  [ / ] LR  - / + sample steps  "
          "s=save ckpt", flush=True)
    trainer = Trainer(cfg)

    plt.ion()
    fig, (ax_grid, ax_loss) = plt.subplots(1, 2, figsize=(12, 6), dpi=105)
    grid_artist = ax_grid.imshow(trainer.grid(), cmap="gray", vmin=0, vmax=1)
    ax_grid.axis("off")
    (loss_line,) = ax_loss.plot([], [], lw=0.8)
    ax_loss.set_yscale("log"); ax_loss.set_xlabel("step"); ax_loss.grid(True, alpha=0.3)
    ax_loss.set_title("loss (not a quality metric)")
    fig.tight_layout()

    def on_key(event) -> None:
        if event.key == "q":
            trainer.quit = True
        elif event.key == "s":
            save(cfg.out, trainer.model, trainer.mcfg, meta={"steps": trainer.step})
            print(f"saved ckpt @ step {trainer.step}", flush=True)
        else:
            trainer.keymap().get(event.key, lambda: None)()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda _e: setattr(trainer, "quit", True))

    t0 = time.perf_counter()
    while not trainer.quit:
        if trainer.paused:
            plt.pause(0.05)
            continue
        loss, grad_norm = trainer.train_step()
        if trainer.step % 25 == 0:
            trainer.history.append((trainer.step, loss))
        if trainer.step % cfg.redraw_every == 0:
            grid_artist.set_data(trainer.grid())
            loss_line.set_data(*zip(*trainer.history))
            ax_loss.relim(); ax_loss.autoscale_view()
            sps = trainer.step / max(time.perf_counter() - t0, 1e-6)
            ax_grid.set_title(
                f"step {trainer.step}  loss {loss:.2f}  gnorm {grad_norm:.1f}  "
                f"{sps:.0f} step/s", fontsize=10)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    save(cfg.out, trainer.model, trainer.mcfg, meta={"steps": trainer.step})
    print(f"saved final ckpt @ step {trainer.step}", flush=True)
    plt.ioff()


if __name__ == "__main__":
    main(tyro.cli(Config))
