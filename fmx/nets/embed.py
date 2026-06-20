import math

import mlx.core as mx
from mlx.core import array


def fourier_time_embed(t: array, n_freqs: int) -> array:
    # t: (B,) in [0, 1] -> (B, 2 * n_freqs); frequencies geometric over 1..1000.
    freqs = mx.exp(mx.linspace(math.log(1.0), math.log(1000.0), n_freqs))
    args = t[:, None] * freqs[None, :]
    return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
