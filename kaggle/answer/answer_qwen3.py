"""Answer the eval questions with Qwen3-8B, base or fine-tuned, on a Kaggle GPU.

Runs as a Kaggle script kernel (see kaggle/README.md). Uses the same system prompt, greedy
decoding and 1,024-token limit as `torchlight answer`, and writes JSONL that `torchlight`
reads as an eval arm.
"""

import json
import subprocess
import sys
import time

ARM = "QF"
MODELS = {"Q": "Qwen/Qwen3-8B", "QF": "samadpls/torchlight-qwen3-8b"}
# Qwen3's training data reportedly runs to March 2025, so only later threads are a fair test.
SINCE = "2025-04-01"
MAX_NEW = 1024
BUDGET = 24_576
SYSTEM = (
    "You are a careful PyTorch helper. Give specific, correct guidance with code where it helps. "
    "If a question lacks the details needed to answer it, ask for them, for example a minimal "
    "executable code snippet or the full error message, instead of guessing."
)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51", "accelerate", "datasets"],
    check=True,
)

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

rows = [r for r in load_dataset("samadpls/torchlight-threads", "eval", split="test") if r["created_at"] >= SINCE]
print(len(rows), "questions since", SINCE, flush=True)

tok = AutoTokenizer.from_pretrained(MODELS[ARM])
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODELS[ARM], dtype=torch.float16, device_map="auto")
model.eval()


def prompt(question):
    chat = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    return tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)


prompts = [prompt(r["question"]) for r in rows]
lengths = [len(tok(p).input_ids) for p in prompts]
order = sorted(range(len(rows)), key=lambda i: lengths[i])

start, i = time.time(), 0
with open(f"/kaggle/working/answers-{ARM}.jsonl", "w") as out:
    while i < len(order):
        # Batches pad to their longest prompt, so long prompts run in smaller batches.
        size = max(1, min(16, BUDGET // (lengths[order[i]] + MAX_NEW)))
        size = max(1, min(size, BUDGET // (max(lengths[j] for j in order[i : i + size]) + MAX_NEW)))
        batch = order[i : i + size]
        enc = tok([prompts[j] for j in batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, seq in zip(batch, gen[:, enc.input_ids.shape[1] :]):
            ids = [t for t in seq.tolist() if t != tok.pad_token_id]
            text = tok.decode(ids, skip_special_tokens=True).strip()
            record = {"arm": ARM, "thread_id": rows[j]["thread_id"], "answer": text, "searches": [], "generated_tokens": len(ids)}
            out.write(json.dumps(record) + "\n")
        out.flush()
        i += len(batch)
        print(f"{i}/{len(rows)} answered, {time.time() - start:.0f} s", flush=True)
print("done", flush=True)
