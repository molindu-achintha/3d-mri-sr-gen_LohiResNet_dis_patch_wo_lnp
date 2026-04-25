from pathlib import Path
from typing import Dict, List


def ensure_header(path: Path, header: List[str]) -> None:
    if path.exists():
        return
    with path.open("w") as handle:
        handle.write("\t".join(header) + "\n")


def append_row(path: Path, values: List[str]) -> None:
    with path.open("a") as handle:
        handle.write("\t".join(values) + "\n")


def read_tabular_log(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}

    with path.open("r") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        return {}

    header = lines[0].split("\t")
    cols: Dict[str, List[float]] = {name: [] for name in header}
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(header):
            continue
        for idx, name in enumerate(header):
            try:
                cols[name].append(float(values[idx]))
            except ValueError:
                cols[name].append(float("nan"))
    return cols


def save_loss_plots(log_dir: Path, iter_log: Path, epoch_log: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not save plots (matplotlib unavailable): {exc}")
        return

    iter_cols = read_tabular_log(iter_log)
    if iter_cols:
        x = iter_cols.get("total_steps")
        if x:
            fig, ax = plt.subplots(figsize=(9, 5))
            for key, values in iter_cols.items():
                if key in {"epoch", "iter", "total_steps"}:
                    continue
                ax.plot(x, values, label=key, linewidth=1.2)
            ax.set_xlabel("Total Steps")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss by Iteration")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(log_dir / "loss_by_iteration.png", dpi=160)
            plt.close(fig)

    epoch_cols = read_tabular_log(epoch_log)
    if epoch_cols:
        x = epoch_cols.get("epoch")
        if x:
            fig, ax = plt.subplots(figsize=(9, 5))
            for key, values in epoch_cols.items():
                if key in {"epoch", "total_steps"}:
                    continue
                ax.plot(x, values, marker="o", label=key, linewidth=1.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")
            ax.set_title("Training Loss by Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(log_dir / "loss_by_epoch.png", dpi=160)
            plt.close(fig)
