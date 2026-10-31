from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from torchlight.train.log import TrainingLog  # noqa: E402

# A dark card like assets/hero.png, so the chart reads the same on light and dark pages.
SURFACE = "#161a22"
GRID = "#2a303b"
TEXT = "#f0efec"
MUTED = "#a9a8a0"
TRAIN = "#d95926"
VALIDATION = "#3987e5"
FONTS = ["Avenir Next", "Helvetica Neue", "DejaVu Sans"]


def _font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next(name for name in FONTS if name in available or name == FONTS[-1])


def smoothed(values: list[float], weight: float) -> list[float]:
    """Exponential moving average, debiased like TensorBoard's. A trailing window would step
    down when the first, highest losses leave it, which looks like a change in training."""
    out, average = [], 0.0
    for step, value in enumerate(values, 1):
        average = weight * average + (1 - weight) * value
        out.append(average / (1 - weight**step))
    return out


def plot_losses(
    log: TrainingLog, path: Path, title: str, subtitle: str, smoothing: float = 0.9
) -> None:
    """Training loss (each report faint, smoothed on top) and validation loss."""
    plt.rcParams["font.family"] = _font()
    batches = [p.batch for p in log.train]
    losses = [p.loss for p in log.train]

    width, height = 10, 5.2
    fig = plt.figure(figsize=(width, height), dpi=200)
    fig.patch.set_alpha(0)
    fig.add_artist(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0,rounding_size=0.02",
            mutation_aspect=width / height,
            transform=fig.transFigure,
            facecolor=SURFACE,
            edgecolor="none",
            zorder=-10,
        )
    )
    ax = fig.add_axes((0.08, 0.13, 0.86, 0.66))
    ax.set_facecolor("none")

    ax.plot(batches, losses, color=TRAIN, alpha=0.28, lw=1, solid_capstyle="round")
    ax.plot(
        batches,
        smoothed(losses, smoothing),
        color=TRAIN,
        lw=2,
        solid_capstyle="round",
        label="Training loss, smoothed",
    )
    val_batches = [p.batch for p in log.validation]
    val_losses = [p.loss for p in log.validation]
    # Points only: the run evaluated every 1,000 batches, and a line between evaluations
    # would draw a trend nobody measured.
    ax.plot(
        val_batches,
        val_losses,
        color=VALIDATION,
        lw=0,
        marker="o",
        ms=9,
        mec=SURFACE,
        mew=2,
        label="Validation loss",
        zorder=5,
    )
    ax.annotate(
        f"{val_losses[-1]:.3f}",
        (val_batches[-1], val_losses[-1]),
        xytext=(8, -14),
        textcoords="offset points",
        color=TEXT,
        fontsize=10,
    )

    right = max(batches[-1], val_batches[-1])
    ax.set_xlim(-0.015 * right, right * 1.04)
    ax.set_ylim(min(min(losses), min(val_losses)) - 0.2, max(losses) + 0.2)
    ax.set_xlabel("Batch (2 chats each)", color=MUTED, fontsize=10)
    ax.set_ylabel("Loss", color=MUTED, fontsize=10)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=TEXT, handlelength=1.6)

    fig.text(0.08, 0.895, title, color=TEXT, fontsize=17, fontweight="bold")
    fig.text(0.08, 0.84, subtitle, color=MUTED, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
