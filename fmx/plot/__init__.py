import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


def reveal(path: Path, *, open_it: bool) -> None:
    print(f"saved {path}")
    if open_it:
        subprocess.run(["open", str(path)], check=False)  # macOS opener


def panel_loss(ax, history: list[tuple[int, float]]) -> None:
    ax.plot([i for i, _ in history], [loss for _, loss in history])
    ax.set_title("loss (not a quality metric)")
    ax.set_xlabel("step")
    ax.set_yscale("log")


def save_fig(fig, out: Path, dpi: int = 110) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
