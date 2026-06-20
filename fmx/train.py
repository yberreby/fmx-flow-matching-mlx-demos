from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.core import array

Sampler = Callable[[int], array]
Couple = Callable[[array, array], tuple[array, array]]


def progress_line(step: int, metrics: dict) -> str:
    return (f"step {step:5d}  loss {metrics['loss']:.4f}  "
            f"gnorm {metrics['grad_norm']:.2f}")


def fm_loss(model: nn.Module, x0: array, x1: array, t: array) -> array:
    tb = t.reshape((t.shape[0],) + (1,) * (x0.ndim - 1))  # (B,) -> (B, 1, ..., 1)
    pred = model(t, (1 - tb) * x0 + tb * x1)
    target = x1 - x0
    per_sample = ((pred - target) ** 2).reshape(x0.shape[0], -1).sum(axis=-1)
    return per_sample.mean()


def make_step(model: nn.Module, opt: optim.Optimizer, grad_clip: float = 1.0):
    loss_and_grad = nn.value_and_grad(model, fm_loss)
    state = [model.state, opt.state]
    max_norm = grad_clip if grad_clip > 0 else float("inf")  # inf reports w/o clipping

    def step(x0: array, x1: array, t: array) -> tuple[array, array]:
        loss, grads = loss_and_grad(model, x0, x1, t)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm)
        opt.update(model, grads)
        return loss, grad_norm

    return mx.compile(step, inputs=state, outputs=state)


def train(model: nn.Module, sampler: Sampler, *, steps: int, batch: int,
          lr: float = 3e-4, grad_clip: float = 1.0, couple: Couple | None = None,
          log_every: int = 200,
          on_log: Callable[[int, dict], None] | None = None,
          ) -> list[tuple[int, float]]:
    opt = optim.Adam(learning_rate=lr)
    step = make_step(model, opt, grad_clip)
    history: list[tuple[int, float]] = []
    for i in range(steps):
        x1 = sampler(batch)
        x0 = mx.random.normal(x1.shape)
        if couple is not None:
            x0, x1 = couple(x0, x1)
        t = mx.random.uniform(0, 1, (batch,))
        loss, grad_norm = step(x0, x1, t)
        if i % log_every == 0 or i == steps - 1:
            metrics = {"loss": loss.item(), "grad_norm": grad_norm.item()}
            history.append((i, metrics["loss"]))
            if on_log is not None:
                on_log(i, metrics)
    return history
