import math
from pathlib import Path

from pydantic import BaseModel


class LoRARun(BaseModel):
    """Hyperparameters of one LoRA run. Defaults follow Unsloth's recommendations."""

    model: str = "microsoft/Phi-4-mini-instruct"
    rank: int = 16
    # mlx-lm takes alpha / rank, so Unsloth's rank 16, alpha 32 is scale 2.0.
    scale: float = 2.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    epochs: float = 1.0
    batch_size: int = 2
    grad_accumulation: int = 4
    warmup: float = 0.05
    max_seq_length: int = 4096
    seed: int = 0

    def iters(self, train_rows: int) -> int:
        return math.ceil(self.epochs * train_rows / self.batch_size)

    def mlx_config(
        self, data_dir: Path, adapter_dir: Path, train_rows: int, iters: int | None = None
    ) -> dict:
        """The config file `mlx_lm lora --config` reads.

        `iters` counts batches, while the learning-rate schedule counts optimizer updates,
        which happen once every `grad_accumulation` batches.
        """
        iters = iters or self.iters(train_rows)
        updates = max(iters // self.grad_accumulation, 1)
        warmup = round(self.warmup * updates)
        return {
            "model": self.model,
            "train": True,
            "data": str(data_dir),
            "adapter_path": str(adapter_dir),
            "fine_tune_type": "lora",
            "num_layers": -1,
            "lora_parameters": {"rank": self.rank, "scale": self.scale, "dropout": 0.0},
            "mask_prompt": True,
            "optimizer": "adamw",
            "optimizer_config": {"adamw": {"weight_decay": self.weight_decay}},
            "learning_rate": self.learning_rate,
            "lr_schedule": {
                "name": "cosine_decay",
                "warmup": warmup,
                "arguments": [self.learning_rate, updates - warmup, 0.0],
            },
            "iters": iters,
            "batch_size": self.batch_size,
            "grad_accumulation_steps": self.grad_accumulation,
            "max_seq_length": self.max_seq_length,
            "grad_checkpoint": True,
            "steps_per_report": 10,
            "steps_per_eval": 500,
            "val_batches": -1,
            "save_every": 500,
            "seed": self.seed,
        }
