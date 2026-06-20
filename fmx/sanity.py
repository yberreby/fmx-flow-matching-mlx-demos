from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array
from mlx.utils import tree_flatten


def weight_report(model: nn.Module) -> dict:
    # tree_flatten is typed list|dict; on a parameter tree it is the (name, array) list.
    params = cast(list[tuple[str, array]], tree_flatten(model.parameters()))
    return {
        "n_params": sum(p.size for _, p in params),
        "n_arrays": len(params),
        "nan": sum(mx.isnan(p).sum().item() for _, p in params),
        "inf": sum(mx.isinf(p).sum().item() for _, p in params),
        "all_zero": [name for name, p in params if p.abs().max().item() == 0.0],
    }


def behavior_report(model: nn.Module, x: array,
                    ts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    # Velocity stats at fixed x across t, plus the two silent-bug checks: a
    # working field must vary with t and with x. A field that ignores t is the
    # canonical flow-matching bug and produces a plausible-but-wrong loss curve.
    b = x.shape[0]
    per_t = {}
    preds = []
    for t in ts:
        v = model(mx.full((b,), t), x)
        preds.append(v)
        per_t[f"t={t:.2f}"] = {
            "mean": v.mean().item(), "std": v.std().item(),
            "absmax": v.abs().max().item(),
            "nan": mx.isnan(v).sum().item(), "inf": mx.isinf(v).sum().item(),
        }
    stack = mx.stack(preds, axis=0)  # (T, B, ...)
    scale = stack.abs().mean().item() + 1e-9
    t_dependence = stack.std(axis=0).mean().item() / scale
    x_dependence = stack.std(axis=1).mean().item() / scale
    return {
        "per_t": per_t,
        "t_dependence": t_dependence,
        "x_dependence": x_dependence,
        "t_conditioning_ok": t_dependence > 0.02,
        "x_conditioning_ok": x_dependence > 0.02,
    }


def report(model: nn.Module, x: array) -> dict:
    return {"weights": weight_report(model), "behavior": behavior_report(model, x)}


def format_report(r: dict) -> str:
    w, b = r["weights"], r["behavior"]
    lines = [
        f"params: {w['n_params']:,} across {w['n_arrays']} arrays"
        f"  | NaN={w['nan']} Inf={w['inf']} | all-zero arrays: {len(w['all_zero'])}",
        f"t-dependence: {b['t_dependence']:.3f}  "
        f"({'OK' if b['t_conditioning_ok'] else 'BROKEN — net ignores t'})",
        f"x-dependence: {b['x_dependence']:.3f}  "
        f"({'OK' if b['x_conditioning_ok'] else 'BROKEN — net ignores x'})",
        "velocity per t:",
    ]
    for t, s in b["per_t"].items():
        lines.append(f"  {t}: mean={s['mean']:+.3f} std={s['std']:.3f} "
                     f"absmax={s['absmax']:.2f} nan={s['nan']} inf={s['inf']}")
    if w["all_zero"]:
        shown = ", ".join(w["all_zero"][:6])
        lines.append(f"all-zero: {shown}" + (" ..." if len(w["all_zero"]) > 6 else ""))
    return "\n".join(lines)
