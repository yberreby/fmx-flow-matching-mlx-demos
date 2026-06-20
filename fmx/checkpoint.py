import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .nets import MLPConfig, UNetConfig

Config = MLPConfig | UNetConfig
_CONFIGS = {cls.__name__: cls for cls in (MLPConfig, UNetConfig)}


def save(path: Path, model: nn.Module, cfg: Config,
         meta: dict | None = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "weights.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(
        {"config": type(cfg).__name__, "fields": asdict(cfg), "meta": meta or {}},
        indent=2))


def load(path: Path) -> tuple[nn.Module, Config, dict]:
    path = Path(path)
    blob = json.loads((path / "config.json").read_text())
    # JSON has no tuple; every sequence field in our configs is a tuple.
    fields = {k: tuple(v) if isinstance(v, list) else v
              for k, v in blob["fields"].items()}
    # fields came from this config's own asdict(); the constructor validates them.
    cfg = _CONFIGS[blob["config"]](**fields)  # pyright: ignore[reportArgumentType]
    model = cfg.build()
    weights = mx.load(str(path / "weights.safetensors"))
    assert isinstance(weights, dict)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model, cfg, blob["meta"]


def latest_ckpt(root: Path = Path("outputs"),
                config_type: str | None = None) -> Path:
    configs = list(Path(root).glob("**/config.json"))
    if config_type is not None:
        configs = [p for p in configs
                   if json.loads(p.read_text())["config"] == config_type]
    if not configs:
        raise FileNotFoundError(f"no {config_type or ''} checkpoints under {root}")
    return max(configs, key=lambda p: p.stat().st_mtime).parent
