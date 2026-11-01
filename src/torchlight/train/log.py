import json
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


def read_trainer_log(path: Path) -> TrainingLog:
    """Training and validation losses from a transformers Trainer's log_history.json, where
    `batch` counts optimizer steps."""
    entries = json.loads(path.read_text())
    return TrainingLog(
        train=[LossPoint(batch=e["step"], loss=e["loss"]) for e in entries if "loss" in e],
        validation=[
            LossPoint(batch=e["step"], loss=e["eval_loss"]) for e in entries if "eval_loss" in e
        ],
    )


def loss_table(
    log: TrainingLog, batches_per_epoch: int, every: int = 500, unit: str = "Batch"
) -> str:
    """A Markdown table of the losses: training loss averaged since the previous row, and
    validation loss wherever the run evaluated."""
    last = max(log.train[-1].batch, *(p.batch for p in log.validation))
    validation = {p.batch: p.loss for p in log.validation}
    rows = sorted({*range(every, last, every), last, *validation})
    lines = [f"| Epoch | {unit} | Training loss | Validation loss |", "|---|---|---|---|"]
    previous = -1
    for batch in rows:
        window = [p.loss for p in log.train if previous < p.batch <= batch]
        train = f"{mean(window):.3f}" if window else "-"
        val = f"{validation[batch]:.3f}" if batch in validation else "-"
        lines.append(f"| {batch / batches_per_epoch:.2f} | {batch:,} | {train} | {val} |")
        previous = batch
    return "\n".join(lines)


class KaggleRun(BaseModel):
    """What a Kaggle training run (kaggle/train_qwen3.py) reports in its log."""

    train_rows: int
    validation_rows: int
    steps: int
    trainable_millions: float
    trainable_percent: float
    total_billions: float
    gpu: str
    unsloth: str
    transformers: str
    hours: float


def _number(text: str) -> int:
    return int(text.replace(",", ""))


def read_kaggle_run(log_path: Path, history_path: Path) -> KaggleRun:
    text = log_path.read_text(errors="replace")

    def find(pattern: str) -> re.Match:
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"{log_path} has no line matching {pattern!r}")
        return match

    sizes = find(r"Num examples = ([\d,]+) \| Num Epochs = \d+ \| Total steps = ([\d,]+)")
    trainable = find(r"Trainable parameters = ([\d,]+) of ([\d,]+) \(([\d.]+)% trained\)")
    versions = find(r"Unsloth (\d+(?:\.\d+)+): Fast \w+ patching\. Transformers: (\d+(?:\.\d+)+)")
    summary = next(e for e in json.loads(history_path.read_text()) if "train_runtime" in e)
    return KaggleRun(
        train_rows=_number(sizes[1]),
        validation_rows=int(find(r"validation: (\d+) chats")[1]),
        steps=_number(sizes[2]),
        trainable_millions=_number(trainable[1]) / 1e6,
        trainable_percent=float(trainable[3]),
        total_billions=_number(trainable[2]) / 1e9,
        gpu=find(r"(Tesla \w+)\. Num GPUs")[1],
        unsloth=versions[1],
        transformers=versions[2],
        hours=summary["train_runtime"] / 3600,
    )
