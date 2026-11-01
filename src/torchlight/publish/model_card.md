---
{front_matter}
---

<p align="center">
  <img alt="torchlight" src="https://huggingface.co/{repo}/resolve/main/assets/hero.png" width="100%">
</p>

<p align="center">
  <a href="https://github.com/samadpls/torchlight"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-samadpls%2Ftorchlight-181717?logo=github"></a>
  <a href="https://huggingface.co/datasets/samadpls/torchlight-threads"><img alt="Dataset" src="https://img.shields.io/badge/dataset-torchlight--threads-FFD21E?logo=huggingface&logoColor=000"></a>
  <a href="https://huggingface.co/{base_model}"><img alt="Base model: Phi-4-mini-instruct" src="https://img.shields.io/badge/base-Phi--4--mini--instruct-555"></a>
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-EE4C2C">
  <a href="https://samadpls.com"><img alt="samadpls.com" src="https://img.shields.io/badge/by-samadpls.com-EE4C2C"></a>
</p>

# {title}

> {variant_note}

## Model Overview

torchlight is a small model that helps with PyTorch questions. It is [{base_name}](https://huggingface.co/{base_model}) ({size} parameters), fine-tuned on {train_rows:,} chats from the [PyTorch forums](https://discuss.pytorch.org). Most end with the answer that solved the thread; the rest teach it to ask for missing details.

It was trained not to guess: when a question has enough detail, suggest a fix; when it doesn't, ask for the code or the full error.

- **Trained on real answers.** Each answer is the post that solved the thread. Most were written by [ptrblck](https://discuss.pytorch.org/u/ptrblck).
- **Tested on new questions.** {eval_count} forum questions that are newer than anything the model saw in training.
- {trained_where}

## Model Details

| | |
|---|---|
| **Base model** | [`{base_model}`](https://huggingface.co/{base_model}) |
| **Size** | {size} parameters ({trainable} trained with LoRA) |
| **Training data** | [`samadpls/torchlight-threads`](https://huggingface.co/datasets/samadpls/torchlight-threads) |
| **Language** | English |
| **License** | {license_name} |
| **Author** | [Abdul Samad](https://samadpls.com) |
| **Code** | [github.com/samadpls/torchlight](https://github.com/samadpls/torchlight) |

The model comes in three versions:

| Repo | What it is | Use it with |
|---|---|---|
| [`{repo_model}`](https://huggingface.co/{repo_model}) | Full model (16-bit) | transformers, mlx-lm |
| [`{repo_lora}`](https://huggingface.co/{repo_lora}) | LoRA adapter only | {lora_runtimes} |
| [`{repo_gguf}`](https://huggingface.co/{repo_gguf}) | GGUF files (Q4_K_M, Q8_0) | llama.cpp, Ollama, LM Studio |

## ⚡ Quickstart

{usage}

Use the same system prompt as in the examples. The model was trained with it.

## 📊 Evaluation

{eval_intro}

{results_table}

{results_notes}

<details>
<summary><b>How the test works</b></summary>

{eval_protocol}

</details>

## Training Details

**Data.** {train_rows:,} chats from forum threads created before {valid_since}. Each chat ends with the answer that solved the thread. {dropped} chats longer than {max_seq_length:,} tokens were left out. The {validation_rows:,} validation chats come from {valid_since} to {eval_since}. The test questions are newer than both.

**Settings.**

| | |
|---|---|
{training_table}

**Training results.**

<p align="center">
  <img alt="Training and validation loss" src="https://huggingface.co/{repo}/resolve/main/assets/training-loss.png" width="100%">
</p>

{loss_table}

{loss_note}

## Intended Use

**Good for:** PyTorch questions about errors, tensor shapes, training loops, data loading and CUDA.

**Not for:** code you run without reading, other frameworks, PyTorch features from after mid-2024, or anything where a wrong answer can cause harm.

## ⚠️ Limitations

{limitations}

## License and Credits

The model is released under the {license_name} license, like the base model. The training data, [`samadpls/torchlight-threads`](https://huggingface.co/datasets/samadpls/torchlight-threads), is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The answers in it were written by the [PyTorch forum](https://discuss.pytorch.org) community, above all [ptrblck](https://discuss.pytorch.org/u/ptrblck), and every row links to its thread. This model is not made or endorsed by PyTorch or the Linux Foundation.

## Citation

```bibtex
@misc{{{cite_key},
  author = {{Abdul Samad}},
  title = {{torchlight: a PyTorch helper fine-tuned on resolved forum threads}},
  year = {{2026}},
  url = {{https://huggingface.co/{repo_model}}}
}}
```

## Contact

Made by **Abdul Samad** ([samadpls.com](https://samadpls.com) · [GitHub @samadpls](https://github.com/samadpls)). Questions or corrections: abdulsamadsid1@gmail.com.
