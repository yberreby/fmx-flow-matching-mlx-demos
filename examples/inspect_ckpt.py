from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import tyro

from fmx.checkpoint import load
from fmx.data.mnist import MNIST_SHAPE
from fmx.nets import UNetConfig
from fmx.sanity import format_report, report


@dataclass
class Config:
    ckpt: Path
    n: int = 64
    seed: int = 0


def main(cfg: Config) -> None:
    mx.random.seed(cfg.seed)
    model, mcfg, meta = load(cfg.ckpt)
    print(f"ckpt: {cfg.ckpt}")
    print(f"config: {mcfg}")
    print(f"meta: {meta}")
    if isinstance(mcfg, UNetConfig):
        x = mx.random.normal((cfg.n, *MNIST_SHAPE[:2], mcfg.channels))
    else:
        x = mx.random.normal((cfg.n, mcfg.dim))
    print(format_report(report(model, x)))


if __name__ == "__main__":
    main(tyro.cli(Config))
