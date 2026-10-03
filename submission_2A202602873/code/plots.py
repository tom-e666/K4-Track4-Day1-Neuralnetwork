import os
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cfg = result["cfg"]
    hist = result["history"]
    epochs = hist["epoch"]
    best_epoch = result["summary"].get("best_epoch", -1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Ô 1: Train & Val Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss", color="#1f77b4", linewidth=1.8)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#d62728", linewidth=1.8)
    if best_epoch > 0:
        axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep ({best_epoch})")
    axes[0].set_title("Loss vs. Epoch", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best", fontsize=9)

    # Ô 2: Val Acc & Val Macro-F1
    axes[1].plot(epochs, hist["val_acc"], label="Val Accuracy", color="#2ca02c", linewidth=1.8)
    if "val_macro_f1" in hist:
        axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#ff7f0e", linewidth=1.8)
    axes[1].axhline(0.4876, color="purple", linestyle=":", alpha=0.6, label="Majority (0.4876)")
    if best_epoch > 0:
        axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_title("Validation Metrics vs. Epoch", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best", fontsize=9)

    # Ô 3: Gradient Norm
    axes[2].plot(epochs, hist["grad_norm"], label="Grad Norm (L2)", color="#9467bd", linewidth=1.8)
    if cfg.get("clip_norm") is not None:
        axes[2].axhline(cfg["clip_norm"], color="red", linestyle="--", alpha=0.7, label=f"Clip {cfg['clip_norm']}")
    axes[2].set_title("Global Gradient Norm (pre-clip)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Norm")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="best", fontsize=9)

    title_str = (
        f"[{cfg.get('exp_id', 'exp')}] {cfg.get('optimizer', 'opt')} lr={cfg.get('lr', 'auto')} "
        f"batch={cfg.get('batch', 512)} loss={cfg.get('loss', 'ce')} | "
        f"Best Val F1={result['summary'].get('val_macro_f1', 0.0):.4f}"
    )
    fig.suptitle(title_str, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "exp")
        epochs = hist["epoch"]
        values = hist.get(metric, [])
        if values:
            ax.plot(epochs, values, label=exp_id, linewidth=1.8)

    ax.set_title(title or f"Comparison: {metric}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
