# Training and answering on Kaggle

The second model, [`torchlight-qwen3-8b`](https://huggingface.co/samadpls/torchlight-qwen3-8b), was trained and tested on Kaggle's free GPUs instead of a Mac. Each folder here is one Kaggle job: a Python script and the `kernel-metadata.json` that tells Kaggle how to run it.

| Folder | What it does | Time on one T4 GPU |
|---|---|---|
| `train/` | Fine-tunes Qwen3-8B with LoRA (Unsloth, 4-bit base), then uploads the adapter, the full model and GGUF files to Hugging Face | about 5 hours |
| `answer/` | Answers the test questions with Qwen3-8B or the fine-tuned model, for the evaluation | 10 to 60 minutes |

## One-time setup

1. Verify your phone number on Kaggle. GPUs and internet access need it.
2. Put your Kaggle API token in `.env` as `KAGGLE_API_TOKEN` (from Kaggle's Settings page, under API).
3. Add your Hugging Face token (with write access) as a Kaggle secret named `HF_API`: open the notebook, click **Edit**, then **Add-ons** → **Secrets**.

`uv run torchlight kaggle ...` runs the `kaggle` command with the token from `.env`.

## Run it

```bash
uv sync --extra kaggle

# 1. Train. Kaggle only gives secrets to runs started from its website, so save the job
#    without running it, then open it and click Save Version → Save & Run All.
uv run torchlight kaggle kernels push -p kaggle/train --no-run

# 2. Answer the test questions. Set ARM in answer_qwen3.py: "Q" for Qwen3-8B, "QF" for the
#    fine-tuned model.
uv run torchlight kaggle kernels push -p kaggle/answer --accelerator NvidiaTeslaT4
uv run torchlight kaggle kernels status samadpls/torchlight-qwen3-8b-answers
uv run torchlight kaggle kernels output samadpls/torchlight-qwen3-8b-answers -p data/eval/qwen/answers
```

Rename the downloaded file to `<ARM>.jsonl` in `data/eval/qwen/answers/`. Then grade and report as in [`src/torchlight/eval/`](../src/torchlight/eval/), on the questions after Qwen3's training data:

```bash
uv run torchlight batches judge --eval-set qwen --since 2025-04-01
uv run torchlight report --eval-set qwen --since 2025-04-01
```

Set `SMOKE = True` in `train_qwen3.py` for a 10-minute test run of every step, without the upload.

## Notes

- **Same recipe as the first model,** except that Unsloth trains on a 4-bit base and puts LoRA on every layer.
- **Qwen3's thinking mode is off** in training and testing, so the model answers directly.
- **Fair test.** Qwen3-8B's training data reportedly runs to March 2025, so it is tested only on the 163 questions asked from April 2025 on.
