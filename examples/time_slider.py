from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("macosx")
import matplotlib.pyplot as plt
import mlx.core as mx
import tyro
from matplotlib.widgets import Slider

from fmx.checkpoint import latest_ckpt, load
from fmx.data.mnist import MNIST_SHAPE
from fmx.nets import UNetConfig
from fmx.plot.image import grid_canvas
from fmx.sample import sample


@dataclass
class Config:
    ckpt: Path | None = None  # default: most recent image checkpoint under outputs/
    k: int = 6
    n_steps: int = 64
    seed: int = 0


def main(cfg: Config) -> None:
    ckpt = cfg.ckpt or latest_ckpt(config_type=UNetConfig.__name__)
    model, mcfg, meta = load(ckpt)
    assert isinstance(mcfg, UNetConfig), "time_slider needs an image checkpoint"
    print(f"loaded {ckpt}  meta={meta}")

    mx.random.seed(cfg.seed)
    n = cfg.k * cfg.k
    traj = sample(model, mx.random.normal((n, *MNIST_SHAPE)), n_steps=cfg.n_steps)
    ts = mx.linspace(0, 1, cfg.n_steps + 1)
    # the model's running guess of the final digit from state x_t: x_t + (1-t) v.
    guess = mx.stack(
        [traj[i] + (1 - ts[i].item()) * model(mx.full((n,), ts[i].item()), traj[i])
         for i in range(cfg.n_steps + 1)], axis=0)
    mx.eval(traj, guess)

    fig, (ax_state, ax_guess) = plt.subplots(1, 2, figsize=(11, 6))
    fig.subplots_adjust(bottom=0.18)
    im_state = ax_state.imshow(grid_canvas(traj[0], cfg.k), cmap="gray", vmin=0, vmax=1)
    im_guess = ax_guess.imshow(grid_canvas(guess[0], cfg.k), cmap="gray", vmin=0, vmax=1)
    ax_state.set_title("x_t (ODE state)"); ax_state.axis("off")
    ax_guess.set_title("predicted final digit"); ax_guess.axis("off")
    slider = Slider(fig.add_axes((0.2, 0.06, 0.6, 0.04)), "t", 0.0, 1.0, valinit=0.0)

    def update(_v) -> None:
        i = int(round(slider.val * cfg.n_steps))
        im_state.set_data(grid_canvas(traj[i], cfg.k))
        im_guess.set_data(grid_canvas(guess[i], cfg.k))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    print("drag t: 0 = noise, 1 = data. close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Config))
