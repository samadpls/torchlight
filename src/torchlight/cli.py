import asyncio
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
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
HUB_REPO = "samadpls/torchlight-phi4-mini"
GGUF_QUANTS = ("Q4_K_M", "Q8_0")
HUB_KINDS = ("model", "lora", "gguf")

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Build the torchlight dataset, train the model and evaluate it."""
    # Downloads and caches stay inside the project, so deleting it frees them.
    data_dir = Settings().data_dir.resolve()
    os.environ.setdefault("HF_HOME", str(data_dir / "hf"))
    os.environ.setdefault("MPLCONFIGDIR", str(data_dir / "matplotlib"))


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
    typer.echo(f"Kaggle token: {'set' if _is_set(s.kaggle_token) else 'missing'}")


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


def _eval_examples(since: datetime | None = None) -> list:
    from torchlight.schemas import EvalExample

    path = Settings().data_dir / "processed" / "eval" / "test.jsonl"
    examples = [EvalExample.model_validate_json(line) for line in path.read_text().splitlines()]
    if since:
        examples = [e for e in examples if e.created_at >= since.replace(tzinfo=UTC)]
    return examples


def _eval_dir(eval_set: str | None) -> Path:
    """data/eval, or a subfolder for a comparison on its own set of questions."""
    root = Settings().data_dir / "eval"
    return root / eval_set if eval_set else root


EvalSet = Annotated[
    str | None, typer.Option(help="Subfolder of data/eval for a separate comparison.")
]
Since = Annotated[
    datetime | None,
    typer.Option(formats=["%Y-%m-%d"], help="Use only questions asked on or after this date."),
]


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


def _all_answers(eval_dir: Path) -> list:
    from torchlight.eval.batches import load_answers
    from torchlight.eval.generate import Answer

    answers = []
    for path in sorted((eval_dir / "answers").glob("*.jsonl")):
        answers += [Answer.model_validate_json(line) for line in path.read_text().splitlines()]
    return answers + load_answers(eval_dir / "agent" / "results", "C")


@app.command()
def batches(
    kind: Annotated[str, typer.Argument(help="questions (for arm C) or judge.")],
    batch_size: int = 20,
    eval_set: EvalSet = None,
    since: Since = None,
) -> None:
    """Write batch files for arm C or the judge, run as LLM agents."""
    from torchlight.build.examples import SYSTEM_PROMPT
    from torchlight.eval.batches import judge_block, question_block, write_batches

    eval_dir = _eval_dir(eval_set)
    examples = _eval_examples(since)
    if kind == "questions":
        out = eval_dir / "agent"
        prompt = Path("prompts/answer.md").read_text().replace("{system_prompt}", SYSTEM_PROMPT)
        out.mkdir(parents=True, exist_ok=True)
        (out / "instructions.md").write_text(prompt)
        count = write_batches([question_block(e) for e in examples], out / "batches", batch_size)
    elif kind == "judge":
        out = eval_dir / "judge"
        by_item: dict[int, dict] = {}
        for a in _all_answers(eval_dir):
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
def report(eval_set: EvalSet = None, since: Since = None) -> None:
    """Print the results table from the answers, the judge's grades and the API check."""
    from torchlight.eval.batches import load_grades
    from torchlight.eval.report import results_table

    eval_dir = _eval_dir(eval_set)
    judge_dir = eval_dir / "judge"
    arms_path = judge_dir / "arms.json"
    arms = (
        {int(k): v for k, v in json.loads(arms_path.read_text()).items()}
        if arms_path.exists()
        else {}
    )
    grades, invalid = load_grades(judge_dir / "results", arms)
    examples = _eval_examples(since)
    ids = {e.thread_id for e in examples}
    answers = [a for a in _all_answers(eval_dir) if a.thread_id in ids]
    table = results_table(examples, answers, grades)
    (eval_dir / "results.md").write_text(table + "\n")
    typer.echo(table)
    typer.echo(f"\n{len(grades)} grades, {invalid} items missing or malformed")


def _lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def _publish_dirs(run_dir: Path) -> dict[str, Path]:
    return {"model": run_dir / "fused", "lora": run_dir / "lora", "gguf": run_dir / "gguf"}


@app.command("export")
def export_run(
    run: Annotated[str, typer.Option(help="Folder of the run under data/train.")] = "phi4-mini",
    repo: Annotated[str, typer.Option(help="Hub repo of the merged model; names GGUF files.")] = (
        HUB_REPO
    ),
    quants: Annotated[
        list[str] | None, typer.Option(help=f"llama.cpp quant types [default: {GGUF_QUANTS}].")
    ] = None,
    llama_cpp: Annotated[
        Path | None, typer.Option(help="A llama.cpp checkout; defaults to data/tools/llama.cpp.")
    ] = None,
) -> None:
    """Tidy a fused run's model files, and write its adapter in PEFT format and its GGUF
    quants, ready to publish."""
    from torchlight.publish.export import gguf_files, match_base_files, peft_adapter

    data_dir = Settings().data_dir
    run_dir = data_dir / "train" / run
    dirs = _publish_dirs(run_dir)
    params = LoRARun.model_validate_json((run_dir / "run.json").read_text())
    match_base_files(dirs["model"], params.model)
    modules = peft_adapter(run_dir / "adapters", dirs["lora"], params)
    _log(f"wrote the PEFT adapter for {modules} modules to {dirs['lora']}")
    paths = gguf_files(
        dirs["model"],
        dirs["gguf"],
        repo.split("/")[-1],
        llama_cpp or data_dir / "tools" / "llama.cpp",
        quants or list(GGUF_QUANTS),
    )
    for path in paths:
        _log(f"wrote {path} ({path.stat().st_size / 1e9:.1f} GB)")


@app.command()
def card(
    run: Annotated[str, typer.Option(help="Folder of the run under data/train.")] = "phi4-mini",
    repo: Annotated[str, typer.Option(help="Hub repo of the merged model.")] = HUB_REPO,
    valid_since: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"])] = datetime(2024, 7, 1),
    notes: Annotated[
        Path | None, typer.Option(help="Markdown to add under the results table.")
    ] = None,
    eval_set: EvalSet = None,
    since: Since = None,
) -> None:
    """Write the model cards and the loss chart for the merged model, adapter and GGUF repos.
    A run trained on Kaggle (kaggle/train/train_qwen3.ipynb) is recognized by its
    log_history.json."""
    from torchlight.publish.card import (
        QWEN3_LIMITATIONS,
        kaggle_run_text,
        mlx_run_text,
        model_card,
        repo_names,
    )
    from torchlight.publish.charts import plot_losses
    from torchlight.train.log import loss_table, read_kaggle_run, read_log, read_trainer_log

    data_dir = Settings().data_dir
    run_dir = data_dir / "train" / run
    params = LoRARun.model_validate_json((run_dir / "run.json").read_text())
    examples = _eval_examples(since)
    clarify = sum(e.needs_clarification for e in examples)
    facts: dict = {
        "valid_since": f"{valid_since:%Y-%m-%d}",
        "eval_since": json.loads((data_dir / "raw" / "topic_ids.json").read_text())["eval_since"],
        "eval_count_since": f"{since or min(e.created_at for e in examples):%Y-%m-%d}",
        "eval_until": f"{max(e.created_at for e in examples):%Y-%m-%d}",
        "eval_count": len(examples),
        "clarify_count": clarify,
        "answerable_count": len(examples) - clarify,
        "search_docs": _lines(data_dir / "retrieval" / "docs.jsonl"),
    }
    sft_rows = _lines(data_dir / "processed" / "sft" / "train.jsonl")
    history = run_dir / "log_history.json"
    if history.exists():
        kaggle = read_kaggle_run(next(run_dir.glob("*.log")), history)
        log = read_trainer_log(history)
        facts |= {
            "train_rows": kaggle.train_rows,
            "dropped": sft_rows - kaggle.train_rows,
            "validation_rows": kaggle.validation_rows,
            "steps": kaggle.steps,
            "gpu": kaggle.gpu,
            "unsloth": kaggle.unsloth,
            "transformers": kaggle.transformers,
            "hours": kaggle.hours,
            "trainable": f"{kaggle.trainable_millions:.1f}M",
            "trainable_share": f"{kaggle.trainable_percent:.2f}%",
            "size": f"{kaggle.total_billions:.1f}B",
        }
        facts |= kaggle_run_text(params, facts, QWEN3_LIMITATIONS)
        table = loss_table(log, kaggle.steps, every=200, unit="Step")
        where, x_label = f"on one {kaggle.gpu} GPU on Kaggle", "Step (8 chats each)"
    else:
        from importlib.metadata import version

        config = yaml.safe_load((run_dir / "lora.yaml").read_text())
        log = read_log(run_dir / "train.log")
        batches = config["iters"]
        started = (run_dir / "lora.yaml").stat().st_mtime
        hours = ((run_dir / "adapters" / "adapters.safetensors").stat().st_mtime - started) / 3600
        train_rows = _lines(run_dir / "data" / "train.jsonl")
        facts |= {
            "train_rows": train_rows,
            "dropped": sft_rows - train_rows,
            "validation_rows": _lines(run_dir / "data" / "valid.jsonl"),
            "batches": batches,
            "warmup_updates": config["lr_schedule"]["warmup"],
            "hours": hours,
            "trainable": f"{log.trainable_millions:.1f}M",
            "trainable_share": f"{log.trainable_percent:.2f}%",
            # trainable millions / (percent / 100) / 1000 gives the total in billions.
            "size": f"{log.trainable_millions / log.trainable_percent / 10:.1f}B",
            "optimizer": {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD"}.get(
                config["optimizer"], config["optimizer"]
            ),
            "mlx_lm": version("mlx-lm"),
            "mlx": version("mlx"),
        }
        facts |= mlx_run_text(params, facts)
        table = loss_table(log, batches)
        where, x_label = "on a MacBook Pro (M3 Pro)", "Batch (2 chats each)"
    chart = Path(f"assets/training-loss-{run}.png")
    plot_losses(
        log,
        chart,
        f"LoRA fine-tune of {params.model.split('/')[-1]}",
        f"One epoch over {facts['train_rows']:,} forum chats {where}",
        x_label=x_label,
    )
    results_path = _eval_dir(eval_set) / "results.md"
    results = (
        results_path.read_text().strip()
        if results_path.exists()
        else "The evaluation is still running. Results will be added here when it finishes."
    )
    hub_gguf = _hub_gguf_sizes(repo_names(repo)["gguf"]) if history.exists() else {}
    for kind, folder in _publish_dirs(run_dir).items():
        folder.mkdir(parents=True, exist_ok=True)
        gguf = {p.name: p.stat().st_size for p in sorted(folder.glob("*.gguf"))} or hub_gguf
        (folder / "README.md").write_text(
            model_card(
                kind,
                repo,
                params,
                facts,
                table,
                results,
                notes.read_text().strip() if notes else "",
                gguf,
            )
        )
        (folder / "assets").mkdir(exist_ok=True)
        shutil.copy("assets/hero.png", folder / "assets" / "hero.png")
        shutil.copy(chart, folder / "assets" / "training-loss.png")
        typer.echo(f"wrote {folder / 'README.md'}")


def _hub_gguf_sizes(repo: str) -> dict[str, int]:
    """GGUF file sizes in a Hub repo, for runs whose files were uploaded from elsewhere."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".gguf")}


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def kaggle(ctx: typer.Context) -> None:
    """Run the kaggle CLI with the token from .env, for example
    `torchlight kaggle kernels push -p kaggle/answer --accelerator NvidiaTeslaT4`."""
    token = Settings().kaggle_token
    if not _is_set(token):
        raise typer.BadParameter("set KAGGLE_API_TOKEN (or KAGGLE_API) in .env")
    env = dict(os.environ, KAGGLE_API_TOKEN=token.get_secret_value())
    command = [str(Path(sys.executable).with_name("kaggle")), *ctx.args]
    raise typer.Exit(subprocess.run(command, env=env).returncode)


@app.command()
def publish(
    run: Annotated[str, typer.Option(help="Folder of the run under data/train.")] = "phi4-mini",
    repo: Annotated[str, typer.Option(help="Hub repo of the merged model.")] = HUB_REPO,
    kinds: Annotated[
        list[str] | None, typer.Option("--kind", help="model, lora or gguf [default: all].")
    ] = None,
) -> None:
    """Upload the merged model, the adapter and the GGUF quants to the Hugging Face Hub."""
    from huggingface_hub import HfApi

    from torchlight.publish.card import repo_names

    settings = Settings()
    if not _is_set(settings.hf_token):
        raise typer.BadParameter("HF_TOKEN is not set in .env")
    api = HfApi(token=settings.hf_token.get_secret_value())
    dirs = _publish_dirs(settings.data_dir / "train" / run)
    for kind in kinds or HUB_KINDS:
        repo_id = repo_names(repo)[kind]
        api.create_repo(repo_id, private=False, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id, folder_path=dirs[kind], commit_message=f"Upload {kind} from torchlight"
        )
        _log(f"uploaded {dirs[kind]} to https://huggingface.co/{repo_id}")
