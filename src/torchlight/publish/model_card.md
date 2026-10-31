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

torchlight is a small model that helps with PyTorch questions. It is [Phi-4-mini-instruct](https://huggingface.co/{base_model}) (3.8B parameters), fine-tuned on {train_rows:,} chats from the [PyTorch forums](https://discuss.pytorch.org). Most end with the answer that solved the thread; the rest teach it to ask for missing details.

It was trained not to guess: when a question has enough detail, suggest a fix; when it doesn't, ask for the code or the full error.

- **Trained on real answers.** Each answer is the post that solved the thread. Most were written by [ptrblck](https://discuss.pytorch.org/u/ptrblck).
- **Tested on new questions.** {eval_count} forum questions that are newer than anything the model saw in training.
- **Trained on a laptop.** {hours:.1f} hours on a MacBook Pro (M3 Pro).

## Model Details

| | |
|---|---|
| **Base model** | [`{base_model}`](https://huggingface.co/{base_model}) |
| **Size** | 3.8B parameters ({trainable} trained with LoRA) |
| **Training data** | [`samadpls/torchlight-threads`](https://huggingface.co/datasets/samadpls/torchlight-threads) |
| **Language** | English |
| **License** | MIT |
| **Author** | [Abdul Samad](https://samadpls.com) |
| **Code** | [github.com/samadpls/torchlight](https://github.com/samadpls/torchlight) |

The model comes in three versions:

| Repo | What it is | Use it with |
|---|---|---|
| [`{repo_model}`](https://huggingface.co/{repo_model}) | Full model (16-bit) | transformers, mlx-lm |
| [`{repo_lora}`](https://huggingface.co/{repo_lora}) | LoRA adapter only | PEFT, mlx-lm |
| [`{repo_gguf}`](https://huggingface.co/{repo_gguf}) | GGUF files (Q4_K_M, Q8_0) | llama.cpp, Ollama, LM Studio |

## ⚡ Quickstart

{usage}

Use the same system prompt as in the examples. The model was trained with it.

## 📊 Evaluation

We asked {eval_count} forum questions from {eval_since} to {eval_until} to four setups. Claude Opus 5.5, acting as judge, scored each answer from 1 to 5 by comparing it with the answer that solved the thread.

{results_table}

{results_notes}

<details>
<summary><b>How the test works</b></summary>

- **Setups.** Phi-4-mini (the base model) and torchlight, each with and without forum search. Search covers {search_docs:,} solved forum threads, up to twice per question.
- **Settings.** The local models use greedy decoding (no sampling) and up to 1,024 new tokens.
- **Scoring.** The judge sees all the answers to a question together, shuffled and without names. It follows [`prompts/judge.md`](https://github.com/samadpls/torchlight/blob/main/prompts/judge.md).
- **Valid torch APIs.** A script checks that every `torch.*` name in an answer's code exists in PyTorch 2.15.
- **Asks for details.** How often an answer asks for more information. This is measured on the {clarify_count} questions where the expert also asked, and on the other {answerable_count}, where asking less is better.
- **Reference answer.** For each question, the judge also scored an answer from Claude Sonnet 5.5 with the same search. It isn't in the table, which compares models of the same size.

</details>

## Training Details

**Data.** {train_rows:,} chats from forum threads created before {valid_since}. Each chat ends with the answer that solved the thread. {dropped} chats longer than {max_seq_length:,} tokens were left out. The {validation_rows:,} validation chats come from {valid_since} to {eval_since}. The test questions are newer than both.

**Settings.**

| | |
|---|---|
| Method | LoRA on the top {layers} of 32 layers; the base weights stay frozen |
| LoRA rank / alpha | {rank} / {alpha:g} |
| Learning rate | {learning_rate}, cosine schedule, {warmup_updates} warmup steps |
| Batch | {batch_size} chats, {grad_accumulation} accumulation steps ({chats_per_update} chats per update) |
| Epochs | 1 ({batches:,} batches) |
| Max length | {max_seq_length:,} tokens |
| Loss | On the final answer only |
| Optimizer | {optimizer}, weight decay {weight_decay:g} |
| Hardware | MacBook Pro (M3 Pro), {hours:.1f} hours |
| Software | mlx-lm {mlx_lm}, MLX {mlx} |

**Training results.**

<p align="center">
  <img alt="Training and validation loss" src="https://huggingface.co/{repo}/resolve/main/assets/training-loss.png" width="100%">
</p>

{loss_table}

Training loss is the average since the previous row. Batch 0 is the base model before training.

## Intended Use

**Good for:** PyTorch questions about errors, tensor shapes, training loops, data loading and CUDA.

**Not for:** code you run without reading, other frameworks, PyTorch features from after mid-2024, or anything where a wrong answer can cause harm.

## ⚠️ Limitations

- It is a small model and can be wrong while sounding sure. Check its code.
- On new forum questions it rarely finds the real fix on its own (see Evaluation). Use it for quick first ideas, not final answers.
- It asks for missing details less often than it should.
- With greedy decoding, about 1 answer in 65 gets stuck repeating a phrase. A repetition penalty or sampling avoids this.
- It learned the forum's style, so its answers are short. Sometimes they are too short to help on their own.
- It knows PyTorch up to mid-2024, not version 2.4 or later.
- The training data leans toward ptrblck's topics: debugging, CUDA, training loops and shape errors.
- English only.

## License and Credits

The model is released under the MIT license, like the base model. The training data, [`samadpls/torchlight-threads`](https://huggingface.co/datasets/samadpls/torchlight-threads), is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The answers in it were written by the [PyTorch forum](https://discuss.pytorch.org) community, above all [ptrblck](https://discuss.pytorch.org/u/ptrblck), and every row links to its thread. This model is not made or endorsed by PyTorch or the Linux Foundation.

## Citation

```bibtex
@misc{{samad2026torchlight_model,
  author = {{Abdul Samad}},
  title = {{torchlight: a PyTorch helper fine-tuned on resolved forum threads}},
  year = {{2026}},
  url = {{https://huggingface.co/{repo_model}}}
}}
```

## Contact

Made by **Abdul Samad** ([samadpls.com](https://samadpls.com) · [GitHub @samadpls](https://github.com/samadpls)). Questions or corrections: abdulsamadsid1@gmail.com.
