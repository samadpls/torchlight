import json
from pathlib import Path

import yaml

from torchlight.build.examples import SYSTEM_PROMPT
from torchlight.train.config import LoRARun

TEMPLATE = Path(__file__).with_name("dataset_card.md")

FILTER_LABELS = {
    "excluded: unresolved": "No accepted answer and no confirmation from the asker",
    "excluded: created before PyTorch 0.4": "Created before PyTorch 0.4 (May 2018)",
    "excluded: question depends on an image": "Question depends on an image",
    "excluded: answer is an image": "Answer is an image",
    "excluded: too long": "Longer than 8 messages or 12,000 characters",
    "excluded: outdated API in answer": "Answer uses removed APIs such as `volatile=True`",
    "excluded: not English": "Not in English",
    "excluded: duplicate question": "Duplicate question",
    "dropped in review": "Dropped by the review agent",
    "composed answer not grounded": "Combined answer whose code was not found in its sources",
    "eval: silver skipped": "Evaluation threads without an accepted answer",
}


def _count(value: int) -> str:
    return f"{value:,} thread" + ("" if value == 1 else "s")


def dataset_card(stats_path: Path, valid_since: str, eval_since: str) -> str:
    stats = json.loads(stats_path.read_text())
    filters = "\n".join(
        f"- {label}: {_count(stats[key])}" for key, label in FILTER_LABELS.items() if stats.get(key)
    )
    splits = ("train", "validation")
    gold = sum(stats.get(f"examples: {s} answer gold", 0) for s in splits)
    silver = sum(stats.get(f"examples: {s} answer silver", 0) for s in splits)
    clarifications = sum(
        stats.get(f"examples: {s} clarification {t}", 0) for s in splits for t in ("gold", "silver")
    )
    usable = sum(stats.get(f"examples: {s}", 0) for s in (*splits, "eval"))
    review = {v: stats.get(f"review: {v}", 0) for v in ("keep", "compose")}
    review_drop = stats.get("dropped in review", 0)
    return TEMPLATE.read_text().format(
        system_prompt=SYSTEM_PROMPT,
        usable=usable,
        usable_badge=f"{usable:,}".replace(",", "%2C"),
        usable_threads=gold + silver + stats.get("examples: eval", 0),
        answers=gold + silver,
        gold_answers=gold,
        silver_answers=silver,
        clarifications=clarifications,
        reviewed=review["keep"] + review["compose"] + review_drop,
        review_keep=review["keep"],
        review_compose=review["compose"],
        review_drop=review_drop,
        threads=sum(v for k, v in stats.items() if k.startswith("threads: ")),
        user_threads=stats.get("source: user topics", 0),
        recent_threads=stats.get("source: other recent solved topics", 0),
        train=stats.get("examples: train", 0),
        validation=stats.get("examples: validation", 0),
        eval=stats.get("examples: eval", 0),
        valid_since=valid_since,
        eval_since=eval_since,
        filters=filters,
    )


MODEL_TEMPLATE = Path(__file__).with_name("model_card.md")
QUESTION = "How do I check which device a tensor is on?"
GGUF_NOTES = {
    "Q4_K_M": "Smaller and faster; a good default",
    "Q8_0": "Closer to the full model; about twice the size",
}


def repo_names(repo: str) -> dict[str, str]:
    return {"model": repo, "lora": f"{repo}-lora", "gguf": f"{repo}-GGUF"}


def _front_matter(kind: str, repos: dict[str, str], base_model: str) -> str:
    meta: dict = {"license": "mit", "language": ["en"]}
    if kind == "gguf":
        meta |= {"base_model": repos["model"], "base_model_relation": "quantized"}
    else:
        meta |= {
            "base_model": base_model,
            "base_model_relation": "adapter" if kind == "lora" else "finetune",
            "library_name": "peft" if kind == "lora" else "transformers",
        }
    meta |= {
        "pipeline_tag": "text-generation",
        "datasets": ["samadpls/torchlight-threads"],
        "tags": ["pytorch", "code", "debugging", "lora", "mlx", "phi3"]
        + (["gguf"] if kind == "gguf" else []),
    }
    return yaml.safe_dump(meta, sort_keys=False).strip()


def _usage(kind: str, repos: dict[str, str], base_model: str, gguf_files: dict[str, int]) -> str:
    mlx_args = (
        f'--max-tokens 512 \\\n  --system-prompt "{SYSTEM_PROMPT}" \\\n  --prompt "{QUESTION}"'
    )
    if kind == "model":
        return f"""### Transformers

```python
from transformers import pipeline

SYSTEM_PROMPT = "{SYSTEM_PROMPT}"

chat = pipeline("text-generation", model="{repos["model"]}", dtype="auto", device_map="auto")
messages = [
    {{"role": "system", "content": SYSTEM_PROMPT}},
    {{"role": "user", "content": "{QUESTION}"}},
]
print(chat(messages, max_new_tokens=512)[0]["generated_text"][-1]["content"])
```

### mlx-lm (Apple Silicon)

```bash
pip install mlx-lm
mlx_lm.generate --model {repos["model"]} {mlx_args}
```"""
    if kind == "lora":
        return f"""### PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{repos["lora"]}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = model.merge_and_unload()  # optional: merge the adapter into the model
```

### mlx-lm (Apple Silicon)

```bash
hf download {repos["lora"]} --include "mlx/*" --local-dir torchlight-lora
mlx_lm.generate --model {base_model} --adapter-path torchlight-lora/mlx {mlx_args}
```"""
    rows = "\n".join(
        f"| `{name}` | {size / 1e9:.1f} GB | {GGUF_NOTES.get(name.rsplit('-', 1)[-1][:-5], '')} |"
        for name, size in gguf_files.items()
    )
    return f"""| File | Size | Notes |
|---|---|---|
{rows}

### llama.cpp

```bash
llama-cli -hf {repos["gguf"]}:Q4_K_M -sys "{SYSTEM_PROMPT}"
```

### Ollama

Set the system prompt with `/set system` once it starts.

```bash
ollama run hf.co/{repos["gguf"]}:Q4_K_M
```"""


def model_card(
    kind: str,
    repo: str,
    params: LoRARun,
    facts: dict,
    loss_table: str,
    results_table: str,
    results_notes: str = "",
    gguf_files: dict[str, int] | None = None,
) -> str:
    repos = repo_names(repo)
    notes = {
        "model": "This repo has the full model, with the fine-tuning merged in.",
        "lora": "This repo has only the LoRA adapter, in PEFT format. The mlx-lm version is in "
        f"`mlx/`. For the full model, see [`{repos['model']}`]"
        f"(https://huggingface.co/{repos['model']}).",
        "gguf": f"This repo has GGUF files of [`{repos['model']}`]"
        f"(https://huggingface.co/{repos['model']}) for llama.cpp, Ollama and LM Studio.",
    }
    return MODEL_TEMPLATE.read_text().format(
        front_matter=_front_matter(kind, repos, params.model),
        repo=repos[kind],
        repo_model=repos["model"],
        repo_lora=repos["lora"],
        repo_gguf=repos["gguf"],
        title=repos[kind].split("/")[-1],
        variant_note=notes[kind],
        usage=_usage(kind, repos, params.model, gguf_files or {}),
        base_model=params.model,
        results_table=results_table,
        results_notes=results_notes,
        loss_table=loss_table,
        rank=params.rank,
        alpha=params.scale * params.rank,
        layers=params.layers,
        learning_rate=f"{params.learning_rate:.0e}".replace("e-0", "e-"),
        weight_decay=params.weight_decay,
        batch_size=params.batch_size,
        grad_accumulation=params.grad_accumulation,
        chats_per_update=params.batch_size * params.grad_accumulation,
        max_seq_length=params.max_seq_length,
        **facts,
    )
