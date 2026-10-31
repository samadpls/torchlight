import re
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# mlx-lm prints "  iter   train_loss   tok/s   tokens" every steps_per_report batches, the
# loss averaged over those batches, and "  iter    val 3.280    266.18s" at each evaluation.
TRAIN_ROW = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s")
VAL_ROW = re.compile(r"^\s*(\d+)\s+val\s+(\d+\.\d+)\s")
TRAINABLE = re.compile(r"Trainable parameters: ([\d.]+)% \(([\d.]+)M/")


class LossPoint(BaseModel):
    batch: int
    loss: float


class TrainingLog(BaseModel):
    train: list[LossPoint]
    validation: list[LossPoint]
    trainable_millions: float | None = None
    trainable_percent: float | None = None


def read_log(path: Path) -> TrainingLog:
    """Training and validation losses from an mlx-lm LoRA log."""
    text = ANSI.sub("", path.read_text(errors="replace")).replace("\r", "\n")
    train, validation = [], []
    trainable = TRAINABLE.search(text)
    for line in text.splitlines():
        if match := VAL_ROW.match(line):
            batch = int(match[1])
            # mlx-lm evaluates before a batch's update, so its first evaluation is the base model.
            validation.append(LossPoint(batch=0 if batch == 1 else batch, loss=float(match[2])))
        elif match := TRAIN_ROW.match(line):
            train.append(LossPoint(batch=int(match[1]), loss=float(match[2])))
    return TrainingLog(
        train=train,
        validation=validation,
        trainable_millions=float(trainable[2]) if trainable else None,
        trainable_percent=float(trainable[1]) if trainable else None,
    )


def loss_table(log: TrainingLog, batches_per_epoch: int, every: int = 500) -> str:
    """A Markdown table of the losses: training loss averaged since the previous row, and
    validation loss wherever the run evaluated."""
    last = log.train[-1].batch
    validation = {p.batch: p.loss for p in log.validation}
    rows = sorted({*range(every, last, every), last, *validation})
    lines = ["| Epoch | Batch | Training loss | Validation loss |", "|---|---|---|---|"]
    previous = -1
    for batch in rows:
        window = [p.loss for p in log.train if previous < p.batch <= batch]
        train = f"{mean(window):.3f}" if window else "-"
        val = f"{validation[batch]:.3f}" if batch in validation else "-"
        lines.append(f"| {batch / batches_per_epoch:.2f} | {batch:,} | {train} | {val} |")
        previous = batch
    return "\n".join(lines)
