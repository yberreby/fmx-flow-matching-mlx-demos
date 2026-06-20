import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from mlx.core import array

_MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist"
_IMAGES = "train-images-idx3-ubyte.gz"

MNIST_SHAPE = (28, 28, 1)  # H, W, C — the single source of truth for image dims.


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def load_mnist(cache: Path = Path("outputs/mnist")) -> array:
    # full training split, scaled to [-1, 1] to sit comparably to N(0, I).
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / _IMAGES
    if not dst.exists():
        urllib.request.urlretrieve(f"{_MIRROR}/{_IMAGES}", dst)
    raw = _read_idx_images(dst).astype(np.float32) / 255.0 * 2.0 - 1.0
    assert raw.shape[1:] == MNIST_SHAPE[:2], raw.shape
    return mx.array(raw[..., None])  # (N, *MNIST_SHAPE)


def batch_sampler(data: array) -> Callable[[int], array]:
    n = data.shape[0]
    return lambda b: data[mx.random.randint(0, n, (b,))]
