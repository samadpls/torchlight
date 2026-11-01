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


def _front_matter(
    kind: str, repos: dict[str, str], base_model: str, license_id: str, tags: list[str]
) -> str:
    meta: dict = {"license": license_id, "language": ["en"]}
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
        "tags": ["pytorch", "code", "debugging", "lora", *tags]
        + (["gguf"] if kind == "gguf" else []),
    }
    return yaml.safe_dump(meta, sort_keys=False).strip()


def _usage(
    kind: str,
    repos: dict[str, str],
    base_model: str,
    gguf_files: dict[str, int],
    mlx_adapter: bool,
    thinking: bool,
) -> str:
    """Quickstart code. `thinking` marks a base model with a thinking mode, which these
    fine-tunes were trained with switched off."""
    mlx_args = (
        f'--max-tokens 512 \\\n  --system-prompt "{SYSTEM_PROMPT}" \\\n  --prompt "{QUESTION}"'
    )
    if thinking:
        mlx_args += " \\\n  --chat-template-config '{\"enable_thinking\": false}'"
    if kind == "model" and thinking:
        return f"""### Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "{SYSTEM_PROMPT}"

repo = "{repos["model"]}"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="auto", device_map="auto")
messages = [
    {{"role": "system", "content": SYSTEM_PROMPT}},
    {{"role": "user", "content": "{QUESTION}"}},
]
# The model was trained with thinking mode off.
text = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### mlx-lm (Apple Silicon)

```bash
pip install mlx-lm
mlx_lm.generate --model {repos["model"]} {mlx_args}
```"""
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
        return (
            f"""### PEFT

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "{repos["lora"]}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = model.merge_and_unload()  # optional: merge the adapter into the model
```"""
            + (
                "\n\nWhen you apply the chat template, pass `enable_thinking=False`: the "
                "adapter was trained with thinking mode off."
                if thinking
                else ""
            )
            + (
                f"""

### mlx-lm (Apple Silicon)

```bash
hf download {repos["lora"]} --include "mlx/*" --local-dir torchlight-lora
mlx_lm.generate --model {base_model} --adapter-path torchlight-lora/mlx {mlx_args}
```"""
                if mlx_adapter
                else ""
            )
        )
    rows = "\n".join(
        f"| `{name}` | {size / 1e9:.1f} GB | {GGUF_NOTES.get(name.rsplit('-', 1)[-1][:-5], '')} |"
        for name, size in gguf_files.items()
    )
    reasoning = " --reasoning off" if thinking else ""
    think = " --think=false" if thinking else ""
    return f"""| File | Size | Notes |
|---|---|---|
{rows}

### llama.cpp

```bash
llama-cli -hf {repos["gguf"]}:Q4_K_M -sys "{SYSTEM_PROMPT}"{reasoning}
```

### Ollama

Set the system prompt with `/set system` once it starts.

```bash
ollama run hf.co/{repos["gguf"]}:Q4_K_M{think}
```"""


BASE_LICENSES = {
    "microsoft/Phi-4-mini-instruct": ("mit", "MIT"),
    "Qwen/Qwen3-8B": ("apache-2.0", "Apache-2.0"),
}
# Base models with a thinking mode; the fine-tunes were trained with it off.
THINKING_MODELS = {"Qwen/Qwen3-8B"}
JUDGE_LINK = (
    "[`prompts/judge.md`](https://github.com/samadpls/torchlight/blob/main/prompts/judge.md)"
)


def settings_table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(f"| {name} | {value} |" for name, value in rows)


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _learning_rate(params: LoRARun) -> str:
    return f"{params.learning_rate:.0e}".replace("e-0", "e-")


def _batch(params: LoRARun) -> str:
    per_update = params.batch_size * params.grad_accumulation
    return (
        f"{params.batch_size} chats, {params.grad_accumulation} accumulation steps "
        f"({per_update} chats per update)"
    )


def mlx_run_text(params: LoRARun, facts: dict) -> dict:
    """Card text for a run trained on a Mac with mlx-lm (`torchlight train`)."""
    rows = [
        ("Method", f"LoRA on the top {params.layers} of 32 layers; the base weights stay frozen"),
        ("LoRA rank / alpha", f"{params.rank} / {params.scale * params.rank:g}"),
        (
            "Learning rate",
            f"{_learning_rate(params)}, cosine schedule, {facts['warmup_updates']} warmup steps",
        ),
        ("Batch", _batch(params)),
        ("Epochs", f"1 ({facts['batches']:,} batches)"),
        ("Max length", f"{params.max_seq_length:,} tokens"),
        ("Loss", "On the final answer only"),
        ("Optimizer", f"{facts['optimizer']}, weight decay {params.weight_decay:g}"),
        ("Hardware", f"MacBook Pro (M3 Pro), {facts['hours']:.1f} hours"),
        ("Software", f"mlx-lm {facts['mlx_lm']}, MLX {facts['mlx']}"),
    ]
    return {
        "tags": ["mlx", "phi3"],
        "mlx_adapter": True,
        "lora_runtimes": "PEFT, mlx-lm",
        "trained_where": f"**Trained on a laptop.** {facts['hours']:.1f} hours on a MacBook Pro "
        "(M3 Pro).",
        "training_table": settings_table(rows),
        "eval_intro": f"We asked {facts['eval_count']} forum questions from {facts['eval_since']} "
        f"to {facts['eval_until']} to four setups. Claude Opus 5.5, acting as judge, scored each "
        "answer from 1 to 5 by comparing it with the answer that solved the thread.",
        "eval_protocol": bullet_list(
            [
                "**Setups.** Phi-4-mini (the base model) and torchlight, each with and without "
                f"forum search. Search covers {facts['search_docs']:,} solved forum threads, up to "
                "twice per question.",
                "**Settings.** The local models use greedy decoding (no sampling) and up to 1,024 "
                "new tokens.",
                "**Scoring.** The judge sees all the answers to a question together, shuffled and "
                f"without names. It follows {JUDGE_LINK}.",
                "**Valid torch APIs.** A script checks that every `torch.*` name in an answer's "
                "code exists in PyTorch 2.15.",
                "**Asks for details.** How often an answer asks for more information. This is "
                f"measured on the {facts['clarify_count']} questions where the expert also asked, "
                f"and on the other {facts['answerable_count']}, where asking less is better.",
                "**Reference answer.** For each question, the judge also scored an answer from "
                "Claude Sonnet 5.5 with the same search. It isn't in the table, which compares "
                "models of the same size.",
            ]
        ),
        "loss_note": "Training loss is the average since the previous row. Batch 0 is the base "
        "model before training.",
        "limitations": bullet_list(
            [
                "It is a small model and can be wrong while sounding sure. Check its code.",
                "On new forum questions it rarely finds the real fix on its own (see Evaluation). "
                "Use it for quick first ideas, not final answers.",
                "It asks for missing details less often than it should.",
                "With greedy decoding, about 1 answer in 65 gets stuck repeating a phrase. A "
                "repetition penalty or sampling avoids this.",
                "It learned the forum's style, so its answers are short. Sometimes they are too "
                "short to help on their own.",
                "It knows PyTorch up to mid-2024, not version 2.4 or later.",
                "The training data leans toward ptrblck's topics: debugging, CUDA, training loops "
                "and shape errors.",
                "English only.",
            ]
        ),
        "cite_key": "samad2026torchlight_model",
    }


QWEN3_LIMITATIONS = [
    "It is a small model and can be wrong while sounding sure. Check its code.",
    "Its answers are short, like the forum's. It gives fewer complete fixes than Qwen3-8B itself "
    "(9% of answers graded 4 or 5, against 17%), which writes much longer answers.",
    "It sometimes asks for details it doesn't need (on 13% of questions that could be answered).",
    "It knows PyTorch up to early 2025.",
    "The training data leans toward ptrblck's topics: debugging, CUDA, training loops and shape "
    "errors.",
    "English only.",
]


def kaggle_run_text(params: LoRARun, facts: dict, limitations: list[str]) -> dict:
    """Card text for a run trained on a Kaggle GPU with Unsloth (kaggle/train_qwen3.py)."""
    rows = [
        (
            "Method",
            "LoRA on every linear layer of all layers; the base weights stay frozen in 4-bit "
            "(QLoRA)",
        ),
        ("LoRA rank / alpha", f"{params.rank} / {params.scale * params.rank:g}"),
        ("Learning rate", f"{_learning_rate(params)}, cosine schedule, 5% warmup"),
        ("Batch", _batch(params)),
        ("Epochs", f"1 ({facts['steps']:,} steps)"),
        ("Max length", f"{params.max_seq_length:,} tokens"),
        ("Loss", "On the final answer only"),
        ("Optimizer", f"AdamW 8-bit, weight decay {params.weight_decay:g}"),
        ("Hardware", f"One {facts['gpu']} GPU on Kaggle, {facts['hours']:.1f} hours"),
        ("Software", f"Unsloth {facts['unsloth']}, Transformers {facts['transformers']}"),
    ]
    return {
        "tags": ["unsloth", "qwen3"],
        "mlx_adapter": False,
        "lora_runtimes": "PEFT",
        "trained_where": f"**Trained for free.** {facts['hours']:.1f} hours on one "
        f"{facts['gpu']} GPU on Kaggle.",
        "training_table": settings_table(rows),
        "eval_intro": f"We asked {facts['eval_count']} forum questions from "
        f"{facts['eval_count_since']} to {facts['eval_until']} to four models. They are newer "
        "than Qwen3-8B's training data, which reportedly runs to March 2025. Claude Opus 5.5, "
        "acting as judge, scored each answer from 1 to 5 by comparing it with the answer that "
        "solved the thread.",
        "eval_protocol": bullet_list(
            [
                "**Models.** Qwen3-8B and this model, plus Phi-4-mini and torchlight-phi4-mini, "
                "the earlier and smaller fine-tune. None of them searches the forum here.",
                "**Settings.** Greedy decoding (no sampling), up to 1,024 new tokens, and Qwen3's "
                "thinking mode off. The Qwen models run in 16-bit on a Kaggle GPU.",
                "**Scoring.** The judge sees all four answers to a question together, shuffled "
                f"and without names. It follows {JUDGE_LINK}.",
                "**Valid torch APIs.** A script checks that every `torch.*` name in an answer's "
                "code exists in PyTorch 2.15.",
                "**Asks for details.** How often an answer asks for more information. This is "
                f"measured on the {facts['clarify_count']} questions where the expert also asked, "
                f"and on the other {facts['answerable_count']}, where asking less is better.",
            ]
        ),
        "loss_note": "Training loss is the average since the previous row. Step 0 is the base "
        "model before training; each step is 8 chats.",
        "limitations": bullet_list(limitations),
        "cite_key": "samad2026torchlight_qwen3",
    }


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
    """`facts` holds the run's numbers plus the text from `mlx_run_text` or
    `kaggle_run_text`."""
    repos = repo_names(repo)
    mlx_note = " The mlx-lm version is in `mlx/`." if facts["mlx_adapter"] else ""
    notes = {
        "model": "This repo has the full model, with the fine-tuning merged in.",
        "lora": f"This repo has only the LoRA adapter, in PEFT format.{mlx_note} For the full "
        f"model, see [`{repos['model']}`](https://huggingface.co/{repos['model']}).",
        "gguf": f"This repo has GGUF files of [`{repos['model']}`]"
        f"(https://huggingface.co/{repos['model']}) for llama.cpp, Ollama and LM Studio.",
    }
    license_id, license_name = BASE_LICENSES[params.model]
    return MODEL_TEMPLATE.read_text().format(
        front_matter=_front_matter(kind, repos, params.model, license_id, facts["tags"]),
        repo=repos[kind],
        repo_model=repos["model"],
        repo_lora=repos["lora"],
        repo_gguf=repos["gguf"],
        title=repos[kind].split("/")[-1],
        variant_note=notes[kind],
        usage=_usage(
            kind,
            repos,
            params.model,
            gguf_files or {},
            facts["mlx_adapter"],
            params.model in THINKING_MODELS,
        ),
        base_model=params.model,
        base_name=params.model.split("/")[-1],
        license_name=license_name,
        results_table=results_table,
        results_notes=results_notes,
        loss_table=loss_table,
        max_seq_length=params.max_seq_length,
        **facts,
    )
