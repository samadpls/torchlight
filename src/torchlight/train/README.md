# Training the model

This folder fine-tunes [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) on the dataset's `sft` split. It uses LoRA and [mlx-lm](https://github.com/ml-explore/mlx-lm), which runs on Apple Silicon. The finished model is on Hugging Face as [`samadpls/torchlight-phi4-mini`](https://huggingface.co/samadpls/torchlight-phi4-mini).

## Run it

```bash
uv sync --all-extras
uv run torchlight train      # fine-tune; about 6.3 hours on a MacBook Pro (M3 Pro)
uv run torchlight fuse       # merge the adapter into the base model
uv run torchlight export     # make the PEFT adapter and the GGUF files
uv run torchlight card       # write the model cards and the loss chart
uv run torchlight publish    # upload the three Hugging Face repos
```

Add `--iters 100` to `train` for a short test run. Run `uv run torchlight train --help` to see every setting.

Everything is saved in `data/train/phi4-mini/`:

| Path | What's inside |
|---|---|
| `lora.yaml`, `run.json` | The settings of the run |
| `data/` | The training and validation chats |
| `adapters/` | The trained adapter, with a checkpoint every 1,000 batches |
| `train.log` | The training log |
| `fused/`, `lora/`, `gguf/` | The three versions of the model, each with its model card |

## Results

Training took 6.3 hours. Validation loss went from 3.280 (the base model) to 1.495.

<p align="center">
  <img alt="Training and validation loss" src="../../../assets/training-loss.png" width="100%">
</p>

| Epoch | Batch | Training loss | Validation loss |
|---|---|---|---|
| 0.00 | 0 | - | 3.280 |
| 0.14 | 500 | 1.995 | - |
| 0.29 | 1,000 | 1.738 | 1.554 |
| 0.43 | 1,500 | 1.713 | - |
| 0.58 | 2,000 | 1.674 | 1.511 |
| 0.72 | 2,500 | 1.665 | - |
| 0.87 | 3,000 | 1.623 | 1.497 |
| 1.00 | 3,459 | 1.687 | 1.495 |

Training loss is the average since the previous row. Batch 0 is the base model before training. Most of the drop happens in the first 1,000 batches.

## Settings

| | |
|---|---|
| Base model | Phi-4-mini-instruct, 3.8B parameters, kept frozen in 16-bit |
| Method | LoRA on the top 16 of 32 layers (all linear layers in them) |
| LoRA rank / alpha | 16 / 32 |
| Trained parameters | 11.5M (0.30% of the model) |
| Learning rate | 2e-4, cosine schedule, 43 warmup steps |
| Batch | 2 chats, 4 accumulation steps (8 chats per update) |
| Epochs | 1 (3,459 batches) |
| Max length | 4,096 tokens. The 7 longer chats were left out, not cut. |
| Loss | Only on the final answer. Earlier replies in a chat may be ideas that didn't work. |
| Optimizer | AdamW, weight decay 0.01 |
| Memory saving | Gradient checkpointing |

## Why these choices

**Why Phi-4-mini.** Its training data ends in June 2024. All 260 test questions are from 2025 or later, so the model hasn't seen them. It also supports tool calls, which the comparison needs. Qwen3-4B was the other option, but its data cutoff isn't published.

**Why 16-bit and only the top 16 layers.** Each setup below trained on the same 30 batches:

| Setup | Seconds per batch | Peak memory | Validation loss after 30 batches |
|---|---|---|---|
| 16-bit, all 32 layers | 9.0 | 15.1 GB | 1.796 |
| 4-bit, all 32 layers | 11.2 | 9.6 GB | 2.097 |
| **16-bit, top 16 layers** | **6.0** | **14.4 GB** | **1.702** |

On a Mac, the 4-bit model uses less memory but trains slower. Training only the top 16 layers was the fastest, and its loss was no worse. All three use gradient checkpointing; without it, the longest chats ran out of memory.

## The three versions

- **Full model** (`fused/`). The adapter merged into the base model. We copy back the base model's tokenizer files, so it also loads in older transformers versions.
- **LoRA adapter** (`lora/`). mlx-lm and PEFT store the two LoRA matrices in different layouts, so `export` converts them. We checked that base + adapter gives the same weights as the full model. The original mlx-lm files are in `mlx/`.
- **GGUF** (`gguf/`). Made with [llama.cpp](https://github.com/ggml-org/llama.cpp), which `export` downloads and builds inside `data/tools/`. Q4_K_M is 2.5 GB and Q8_0 is 4.1 GB.

## Training on an NVIDIA GPU

[`notebooks/train_unsloth.ipynb`](../../../notebooks/train_unsloth.ipynb) does the same training with [Unsloth](https://github.com/unslothai/unsloth) on a CUDA GPU, for example in Google Colab. It hasn't been run to the end yet. The published model comes from the Mac run above.

## Files

| File | What it does |
|---|---|
| `config.py` | The training settings, and the config file mlx-lm reads |
| `data.py` | Writes the chats in mlx-lm's format, with the system prompt first |
| `log.py` | Reads the losses from the training log |
| [`publish/export.py`](../publish/export.py) | Makes the PEFT adapter and the GGUF files |
| [`publish/card.py`](../publish/card.py) | Writes the model cards |
| [`publish/charts.py`](../publish/charts.py) | Draws the loss chart |
