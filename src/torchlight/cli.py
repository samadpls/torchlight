import asyncio
import json
import subprocess
import sys
from datetime import datetime
from typing import Annotated

import typer
import yaml
from pydantic import SecretStr

from torchlight import scrape
from torchlight.build.pipeline import build as build_dataset
from torchlight.publish.card import dataset_card
from torchlight.settings import Settings
from torchlight.train import LoRARun, export

DEFAULT_RUN = LoRARun()

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Build the torchlight dataset, train the model and evaluate it."""


def _is_set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value())


def _log(message: str) -> None:
    typer.echo(f"{datetime.now():%H:%M:%S} {message}")


@app.command()
def settings() -> None:
    """Show resolved settings. Secrets are reported only as set or missing."""
    s = Settings()
    typer.echo(f"data_dir: {s.data_dir.resolve()}")
    typer.echo(f"HF token: {'set' if _is_set(s.hf_token) else 'missing'}")
    typer.echo(f"Anthropic API key: {'set' if _is_set(s.anthropic_api_key) else 'missing'}")


@app.command()
def crawl(
    username: str = "ptrblck",
    eval_since: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])] = datetime(2025, 1, 1),
    rate: Annotated[float, typer.Option(help="Requests per second.")] = 3.0,
    concurrency: int = 8,
    limit: Annotated[int | None, typer.Option(help="Fetch at most this many topics.")] = None,
) -> None:
    """Cache every topic the user posted in, plus recent solved topics for evaluation."""
    raw_dir = Settings().data_dir / "raw"
    asyncio.run(scrape.crawl(raw_dir, username, eval_since.date(), rate, concurrency, limit, _log))


@app.command()
def build(
    valid_since: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])] = datetime(2024, 7, 1),
    batch_size: Annotated[int, typer.Option(help="Threads per review batch.")] = 80,
    review_eval: Annotated[bool, typer.Option(help="Review every eval answer.")] = True,
) -> None:
    """Build the dataset files from the crawl cache and write batches still needing review."""
    data_dir = Settings().data_dir
    stats, batch_count = build_dataset(data_dir, valid_since.date(), batch_size, review_eval)
    eval_since = json.loads((data_dir / "raw" / "topic_ids.json").read_text())["eval_since"]
    (data_dir / "processed" / "README.md").write_text(
        dataset_card(data_dir / "processed" / "stats.json", f"{valid_since:%Y-%m-%d}", eval_since)
    )
    for key, value in sorted(stats.items()):
        typer.echo(f"{key}: {value}")
    typer.echo(f"review batches written: {batch_count}")


def _mlx_lm(*args: str) -> None:
    result = subprocess.run([sys.executable, "-m", "mlx_lm", *args])
    if result.returncode:
        raise typer.Exit(result.returncode)


@app.command()
def train(
    run: Annotated[str, typer.Option(help="Folder for this run under data/train.")] = "phi4-mini",
    model: str = DEFAULT_RUN.model,
    rank: int = DEFAULT_RUN.rank,
    scale: Annotated[float, typer.Option(help="LoRA alpha divided by rank.")] = DEFAULT_RUN.scale,
    learning_rate: float = DEFAULT_RUN.learning_rate,
    epochs: float = DEFAULT_RUN.epochs,
    batch_size: int = DEFAULT_RUN.batch_size,
    grad_accumulation: int = DEFAULT_RUN.grad_accumulation,
    warmup: Annotated[float, typer.Option(help="Share of updates spent warming up.")] = (
        DEFAULT_RUN.warmup
    ),
    max_seq_length: int = DEFAULT_RUN.max_seq_length,
    iters: Annotated[
        int | None, typer.Option(help="Stop after this many batches, for trial runs.")
    ] = None,
) -> None:
    """Fine-tune the base model with LoRA on the sft split, using mlx-lm."""
    from transformers import AutoTokenizer

    params = LoRARun(
        model=model,
        rank=rank,
        scale=scale,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        grad_accumulation=grad_accumulation,
        warmup=warmup,
        max_seq_length=max_seq_length,
    )
    data_dir = Settings().data_dir
    run_dir = data_dir / "train" / run
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    stats = export(
        data_dir / "processed" / "sft",
        run_dir / "data",
        lambda messages: len(tokenizer.apply_chat_template(messages, return_dict=False)),
        params.max_seq_length,
    )
    for key, value in stats.items():
        typer.echo(f"{key}: {value}")
    config = params.mlx_config(run_dir / "data", run_dir / "adapters", stats["train: kept"], iters)
    config_path = run_dir / "lora.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    _log(f"training for {config['iters']} batches with {config_path}")
    _mlx_lm("lora", "--config", str(config_path))


@app.command()
def fuse(
    run: Annotated[str, typer.Option(help="Folder of the run under data/train.")] = "phi4-mini",
) -> None:
    """Merge a run's LoRA adapter into its base model and save a Hugging Face folder."""
    run_dir = Settings().data_dir / "train" / run
    config = yaml.safe_load((run_dir / "lora.yaml").read_text())
    _mlx_lm(
        "fuse",
        "--model",
        config["model"],
        "--adapter-path",
        config["adapter_path"],
        "--save-path",
        str(run_dir / "fused"),
    )
