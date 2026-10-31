---
license: cc-by-4.0
language:
- en
pretty_name: torchlight threads
task_categories:
- question-answering
- text-generation
tags:
- pytorch
- code
- sft
- forum
- debugging
size_categories:
- 10K<n<100K
configs:
- config_name: sft
  default: true
  data_files:
  - split: train
    path: sft/train.jsonl
  - split: validation
    path: sft/validation.jsonl
- config_name: eval
  data_files:
  - split: test
    path: eval/test.jsonl
- config_name: threads
  data_files:
  - split: train
    path: threads/threads.jsonl
---

<p align="center">
  <img alt="torchlight" src="https://huggingface.co/datasets/samadpls/torchlight-threads/resolve/main/assets/hero.png" width="100%">
</p>

<p align="center"><a href="https://github.com/samadpls/torchlight"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-samadpls%2Ftorchlight-181717?logo=github"></a> <a href="https://samadpls.com"><img alt="samadpls.com" src="https://img.shields.io/badge/by-samadpls.com-EE4C2C"></a> <img alt="License: CC BY 4.0" src="https://img.shields.io/badge/license-CC%20BY%204.0-EE4C2C"> <img alt="Language: English" src="https://img.shields.io/badge/language-English-555"></p>

# 🔥 torchlight threads

> PyTorch questions from the [PyTorch forums](https://discuss.pytorch.org), each paired with **the answer that actually resolved it**, whoever wrote it.

The threads are the {user_threads:,} topics that [ptrblck](https://discuss.pytorch.org/u/ptrblck), the forum's most active answerer, took part in, plus {recent_threads:,} recent solved topics from other answerers for evaluation. Which answer counts comes from the forum's own signals, not from who wrote it: the accepted solution, or the post the asker said fixed their problem.

The dataset was built to fine-tune and evaluate **torchlight**, a small PyTorch helper that gives a specific fix when it can and asks for a minimal repro when it can't.

## At a glance

| | |
|---|---|
| Threads crawled and cleaned | **{threads:,}** |
| Usable rows (`sft` + `eval`) | **{usable:,}**, from {usable_threads:,} threads |
| Answer examples | {answers:,} ({gold_answers:,} gold, {silver_answers:,} silver) |
| Clarification examples | {clarifications:,} |
| Evaluation questions | {eval:,}, all gold and all reviewed |
| Threads reviewed by an agent | {reviewed:,} ({review_keep:,} kept, {review_compose:,} combined, {review_drop:,} dropped) |

## ⚡ Quick start

```python
from datasets import load_dataset

sft = load_dataset("samadpls/torchlight-threads", "sft")
evaluation = load_dataset("samadpls/torchlight-threads", "eval", split="test")
threads = load_dataset("samadpls/torchlight-threads", "threads", split="train")

print(sft["train"][0]["messages"])
```

## Configs

| Config | Split | Rows | What it holds |
|---|---|---|---|
| `sft` | train | {train:,} | Chat examples (`messages`) for supervised fine-tuning, threads created before {valid_since} |
| `sft` | validation | {validation:,} | Same format, threads from {valid_since} to {eval_since} |
| `eval` | test | {eval:,} | Questions with a reviewed reference answer, threads created on or after {eval_since} |
| `threads` | train | {threads:,} | Every crawled thread, cleaned, with its resolution and the reason it was excluded, if it was |

The splits are by date, never random, so every evaluation question is newer than anything in training.

## How answers were chosen

1. **Accepted solution.** If a post is marked as the solution, it is the answer. When the asker marked their own thank-you post, the answer is the post they were thanking.
2. **Asker confirmation.** Otherwise, if the asker later said the fix worked ("that solved it", "works now"), the answer is the post they were replying to.
3. **Review.** A small LLM agent following [`prompts/review.md`](https://github.com/samadpls/torchlight/blob/main/prompts/review.md) reviewed every evaluation answer, plus training answers that were only a question, under 60 characters, mostly a link, or pointing at other posts. It kept the post, dropped the thread, or combined several posts into one answer.
4. **Grounding check.** A combined answer is rejected unless every code block in it appears verbatim in its source posts.

**Tiers:** `gold` answers are accepted solutions; `silver` answers were confirmed by the asker. The `eval` config is gold only.

## What a row looks like

<details>
<summary><b>An <code>sft</code> answer example</b></summary>

```
user       How does torchvision.transforms.Normalize work
           Whenever I Normalize my input images with its mean and std, and when i plot the
           images to visualize i get this message along with plots of images which are
           distorted in a way. Clipping input data to the valid range ...
assistant  The easiest way would be to plot them before normalizing.
           However, if that's not possible, you could also undo the normalization:
           (code that reverses Normalize with the same mean and std)
```

Each row also carries `kind`, `tier`, `answer_origin`, `answer_post_numbers`, `answer_authors`, `thread_id`, `url` and `created_at`.
</details>

<details>
<summary><b>The two kinds of <code>sft</code> examples</b></summary>

- `answer`: the conversation up to the answer, ending with the answer as the final assistant turn. Earlier assistant turns are context only.
- `clarification`: the opening question and a first reply that asked for missing details, such as a minimal executable snippet or the full error. These teach the model to ask instead of guessing.
</details>

### System prompt

Rows contain only `user` and `assistant` turns, so you can train with any system prompt. torchlight trains with this one:

> {system_prompt}

```python
SYSTEM_PROMPT = "You are a careful PyTorch helper. ..."
sft = sft.map(
    lambda row: {{"messages": [{{"role": "system", "content": SYSTEM_PROMPT}}] + row["messages"]}}
)
```

## Cleaning and filters

Every thread in all three configs is cleaned:

- text quoted from earlier posts is removed;
- images become `[image]`;
- emails and strings that look like API keys are redacted;
- deleted and hidden posts are dropped;
- code blocks are kept exactly as written.

{usable_threads:,} of the {threads:,} threads became examples. The rest stay in `threads` with the reason in `excluded`:

{filters}

## Fields

<details>
<summary><b>Show all fields</b></summary>

| Config | Fields |
|---|---|
| `sft` | `messages`, `kind`, `tier`, `answer_origin` (`verbatim` or `composed`), `answer_post_numbers`, `answer_authors`, `thread_id`, `url`, `created_at` |
| `eval` | `question`, `reference_answer`, `needs_clarification`, `answered_by_ptrblck`, `tier`, `answer_origin`, `answer_post_numbers`, `answer_authors`, `thread_id`, `url`, `created_at` |
| `threads` | `thread_id`, `url`, `title`, `created_at`, `category_id`, `tags`, `asker`, `split`, `resolution`, `excluded`, `posts` |
</details>

## ⚠️ Limitations

- Answers are only as correct as the forum's resolution signals. An accepted or confirmed answer can still be imperfect or tied to an older PyTorch version.
- The data leans toward the questions ptrblck answered, so it is heavy on debugging, CUDA, training loops and shape errors.
- Threads whose question or answer depends on an image were excluded.

## License and attribution

Forum content is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by the [LF Projects Terms of Use](https://lfprojects.org/policies/terms-of-use/) that the PyTorch forums follow, and this dataset is released under the same license. Credit goes to the forum's authors; every row carries the URL of the thread it came from.

## Author

Built by **Abdul Samad** ([samadpls.com](https://samadpls.com) · [GitHub @samadpls](https://github.com/samadpls)) with [torchlight](https://github.com/samadpls/torchlight). Questions and corrections: abdulsamadsid1@gmail.com.

## Citation

```bibtex
@misc{{samad2026torchlight,
  author = {{Abdul Samad}},
  title = {{torchlight threads: resolved PyTorch forum questions and answers}},
  year = {{2026}},
  url = {{https://huggingface.co/datasets/samadpls/torchlight-threads}}
}}
```
