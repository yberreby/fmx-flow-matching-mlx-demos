from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array

from .embed import fourier_time_embed


class TimeEmbed(nn.Module):
    def __init__(self, n_freqs: int, dim: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.l1 = nn.Linear(2 * n_freqs, dim)
        self.l2 = nn.Linear(dim, dim)

    def __call__(self, t: array) -> array:
        return self.l2(nn.silu(self.l1(fourier_time_embed(t, self.n_freqs))))


class ResBlock(nn.Module):
    # GroupNorm-SiLU-Conv twice, with an adaLN scale+shift from the time
    # embedding injected after the second norm. NHWC throughout.
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, n_groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(n_groups, in_ch), in_ch, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(n_groups, out_ch), out_ch, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(t_dim, 2 * out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def __call__(self, x: array, temb: array) -> array:
        h = self.conv1(nn.silu(self.norm1(x)))
        scale, shift = mx.split(self.emb(temb)[:, None, None, :], 2, axis=-1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(nn.silu(h))
        return h + self.skip(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def __call__(self, x: array) -> array:
        return self.conv(self.up(x))


class VelocityUNet(nn.Module):
    def __init__(self, cfg: "UNetConfig"):
        super().__init__()
        self.time_embed = TimeEmbed(cfg.n_time_freqs, cfg.t_dim)
        self.stem = nn.Conv2d(cfg.channels, cfg.base, 3, padding=1)

        widths = [cfg.base * m for m in cfg.ch_mults]
        self.down_blocks: list[ResBlock] = []
        self.downsamples: list[nn.Module] = []
        prev = cfg.base
        for w in widths:
            self.down_blocks.append(ResBlock(prev, w, cfg.t_dim, cfg.n_groups))
            self.downsamples.append(nn.Conv2d(w, w, 3, stride=2, padding=1))
            prev = w

        self.mid = ResBlock(prev, prev, cfg.t_dim, cfg.n_groups)

        self.upsamples: list[nn.Module] = []
        self.up_blocks: list[ResBlock] = []
        for w in reversed(widths):
            self.upsamples.append(Upsample(prev))
            self.up_blocks.append(ResBlock(prev + w, w, cfg.t_dim, cfg.n_groups))  # +w skip
            prev = w

        self.out_norm = nn.GroupNorm(min(cfg.n_groups, prev), prev, pytorch_compatible=True)
        self.out_conv = nn.Conv2d(prev, cfg.channels, 3, padding=1)

    def __call__(self, t: array, x: array) -> array:
        temb = self.time_embed(t)
        h = self.stem(x)
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, temb)
            skips.append(h)
            h = down(h)
        h = self.mid(h, temb)
        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = block(mx.concatenate([up(h), skip], axis=-1), temb)
        return self.out_conv(nn.silu(self.out_norm(h)))


@dataclass(frozen=True)
class UNetConfig:
    channels: int = 1
    base: int = 32
    ch_mults: tuple[int, ...] = (1, 2)
    n_time_freqs: int = 16
    t_dim: int = 128
    n_groups: int = 8

    def build(self) -> VelocityUNet:
        return VelocityUNet(self)
