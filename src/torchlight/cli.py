import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
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
    # Downloads and caches stay inside the project, so deleting it frees them.
    os.environ.setdefault("HF_HOME", str(Settings().data_dir.resolve() / "hf"))


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


def _training_weights(params: LoRARun, models_dir: Path) -> str:
    if params.bits is None:
        return params.model
    path = models_dir / f"{params.model.split('/')[-1]}-{params.bits}bit"
    if not path.exists():
        _log(f"quantizing {params.model} to {params.bits} bits")
        _mlx_lm(
            "convert",
            "--model",
            params.model,
            "-q",
            "--q-bits",
            str(params.bits),
            "--q-group-size",
            "64",
            "--mlx-path",
            str(path),
        )
    return str(path)


@app.command()
def train(
    run: Annotated[str, typer.Option(help="Folder for this run under data/train.")] = "phi4-mini",
    model: str = DEFAULT_RUN.model,
    bits: Annotated[
        int | None, typer.Option(help="Quantize the frozen base for training; 0 keeps 16-bit.")
    ] = DEFAULT_RUN.bits,
    layers: Annotated[
        int, typer.Option(help="Add LoRA to this many top blocks; -1 for all.")
    ] = DEFAULT_RUN.layers,
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
    grad_checkpoint: Annotated[
        bool,
        typer.Option(help="Recompute activations to save memory, at about a third more compute."),
    ] = DEFAULT_RUN.grad_checkpoint,
    iters: Annotated[
        int | None, typer.Option(help="Stop after this many batches, for trial runs.")
    ] = None,
) -> None:
    """Fine-tune the base model with LoRA on the sft split, using mlx-lm."""
    from transformers import AutoTokenizer

    params = LoRARun(
        model=model,
        bits=bits or None,
        layers=layers,
        rank=rank,
        scale=scale,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        grad_accumulation=grad_accumulation,
        warmup=warmup,
        max_seq_length=max_seq_length,
        grad_checkpoint=grad_checkpoint,
    )
    data_dir = Settings().data_dir
    run_dir = data_dir / "train" / run
    weights = _training_weights(params, data_dir / "models")
    tokenizer = AutoTokenizer.from_pretrained(params.model)
    stats = export(
        data_dir / "processed" / "sft",
        run_dir / "data",
        lambda messages: len(tokenizer.apply_chat_template(messages, return_dict=False)),
        params.max_seq_length,
    )
    for key, value in stats.items():
        typer.echo(f"{key}: {value}")
    config = params.mlx_config(
        weights, run_dir / "data", run_dir / "adapters", stats["train: kept"], iters
    )
    config_path = run_dir / "lora.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "run.json").write_text(params.model_dump_json(indent=2))
    _log(f"training for {config['iters']} batches with {config_path}")
    _mlx_lm("lora", "--config", str(config_path))


@app.command()
def fuse(
    run: Annotated[str, typer.Option(help="Folder of the run under data/train.")] = "phi4-mini",
) -> None:
    """Merge a run's adapter into the 16-bit base model and save a Hugging Face folder."""
    run_dir = Settings().data_dir / "train" / run
    params = LoRARun.model_validate_json((run_dir / "run.json").read_text())
    _mlx_lm(
        "fuse",
        "--model",
        params.model,
        "--adapter-path",
        str(run_dir / "adapters"),
        "--save-path",
        str(run_dir / "fused"),
    )


LOCAL_ARMS = {"A": (False, False), "B": (False, True), "D": (True, False), "E": (True, True)}


def _eval_examples() -> list:
    from torchlight.schemas import EvalExample

    path = Settings().data_dir / "processed" / "eval" / "test.jsonl"
    return [EvalExample.model_validate_json(line) for line in path.read_text().splitlines()]


@app.command()
def index() -> None:
    """Build the search_forum index over resolved training threads."""
    from torchlight.retrieval import build_index, load_docs

    data_dir = Settings().data_dir
    docs = load_docs(data_dir / "processed" / "threads" / "threads.jsonl")
    build_index(docs, data_dir / "retrieval")
    typer.echo(f"indexed {len(docs)} threads")


@app.command()
def search(query: str, k: int = 3) -> None:
    """Search resolved training threads, as the search_forum tool does."""
    from torchlight.retrieval import ForumSearch, format_hits

    typer.echo(format_hits(ForumSearch(Settings().data_dir / "retrieval").search(query, k)))


@app.command()
def answer(
    arm: Annotated[
        str, typer.Option(help="A or B for the base model, D or E for the fine-tuned one.")
    ],
    run: Annotated[str, typer.Option(help="Training run whose fused model D and E use.")] = (
        "phi4-mini"
    ),
    max_tokens: int = 1024,
    limit: Annotated[int | None, typer.Option(help="Answer only the first N questions.")] = None,
) -> None:
    """Answer the eval questions with a local model and save them for judging."""
    import time

    from mlx_lm import load

    from torchlight.eval.generate import answer_all
    from torchlight.retrieval import ForumSearch

    fine_tuned, with_tool = LOCAL_ARMS[arm]
    data_dir = Settings().data_dir
    model_path = str(data_dir / "train" / run / "fused") if fine_tuned else DEFAULT_RUN.model
    examples = _eval_examples()[:limit]
    model, tokenizer = load(model_path)
    search_tool = ForumSearch(data_dir / "retrieval") if with_tool else None
    start = time.time()
    answers = answer_all(model, tokenizer, arm, examples, search_tool, max_tokens)
    seconds = time.time() - start
    out = data_dir / "eval" / "answers" / f"{arm}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(a.model_dump_json() + "\n" for a in answers))
    tokens = sum(a.generated_tokens for a in answers)
    searched = sum(bool(a.searches) for a in answers)
    _log(
        f"arm {arm}: {len(answers)} answers in {seconds / 60:.1f} min, {tokens} tokens, "
        f"{searched} searched; wrote {out}"
    )


def _all_answers(data_dir: Path) -> list:
    from torchlight.eval.batches import load_answers
    from torchlight.eval.generate import Answer

    answers = []
    for path in sorted((data_dir / "eval" / "answers").glob("*.jsonl")):
        answers += [Answer.model_validate_json(line) for line in path.read_text().splitlines()]
    return answers + load_answers(data_dir / "eval" / "agent" / "results", "C")


@app.command()
def batches(
    kind: Annotated[str, typer.Argument(help="questions (for arm C) or judge.")],
    batch_size: int = 20,
) -> None:
    """Write batch files for arm C or the judge, run as LLM agents."""
    from torchlight.build.examples import SYSTEM_PROMPT
    from torchlight.eval.batches import judge_block, question_block, write_batches

    data_dir = Settings().data_dir
    examples = _eval_examples()
    if kind == "questions":
        out = data_dir / "eval" / "agent"
        prompt = Path("prompts/answer.md").read_text().replace("{system_prompt}", SYSTEM_PROMPT)
        out.mkdir(parents=True, exist_ok=True)
        (out / "instructions.md").write_text(prompt)
        count = write_batches([question_block(e) for e in examples], out / "batches", batch_size)
    elif kind == "judge":
        out = data_dir / "eval" / "judge"
        by_item: dict[int, dict] = {}
        for a in _all_answers(data_dir):
            by_item.setdefault(a.thread_id, {})[a.arm] = a
        blocks = [judge_block(e, by_item[e.thread_id]) for e in examples if e.thread_id in by_item]
        out.mkdir(parents=True, exist_ok=True)
        arms = {tid: sorted(answers) for tid, answers in by_item.items()}
        (out / "arms.json").write_text(json.dumps(arms))
        count = write_batches(blocks, out / "batches", batch_size)
    else:
        raise typer.BadParameter("kind must be questions or judge")
    typer.echo(f"wrote {count} batches to {out / 'batches'}")


@app.command()
def report() -> None:
    """Print the results table from the answers, the judge's grades and the API check."""
    from torchlight.eval.batches import load_grades
    from torchlight.eval.report import results_table

    data_dir = Settings().data_dir
    judge_dir = data_dir / "eval" / "judge"
    arms_path = judge_dir / "arms.json"
    arms = (
        {int(k): v for k, v in json.loads(arms_path.read_text()).items()}
        if arms_path.exists()
        else {}
    )
    grades, invalid = load_grades(judge_dir / "results", arms)
    table = results_table(_eval_examples(), _all_answers(data_dir), grades)
    (data_dir / "eval" / "results.md").write_text(table + "\n")
    typer.echo(table)
    typer.echo(f"\n{len(grades)} grades, {invalid} items missing or malformed")
