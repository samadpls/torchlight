import asyncio
import json
from datetime import datetime
from typing import Annotated

import typer
from pydantic import SecretStr

from torchlight import scrape
from torchlight.build.pipeline import build as build_dataset
from torchlight.publish.card import dataset_card
from torchlight.settings import Settings

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
