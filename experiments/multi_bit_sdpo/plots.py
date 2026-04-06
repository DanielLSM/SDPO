from __future__ import annotations

import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def plot_kl_trajectory(run_dir: str | Path, seed: int = 0, output_path: str | Path | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting multi_bit_sdpo results") from exc

    run_dir = Path(run_dir)
    records = _load_jsonl(run_dir / f"seed_{seed:04d}.jsonl")
    xs = [record["step"] for record in records]
    ys = [record["kl_to_p_star_after"] for record in records]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, label="KL(p* || pi_k)")
    ax.set_xlabel("Step")
    ax.set_ylabel("KL to p*")
    ax.set_title("multi_bit_sdpo KL trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path is None:
        output_path = run_dir / f"seed_{seed:04d}_kl.png"
    output_path = Path(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
