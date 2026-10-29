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

# torchlight threads

PyTorch questions from the [PyTorch forums](https://discuss.pytorch.org), each paired with the answer that actually resolved it. It was built to fine-tune and evaluate torchlight, a small PyTorch helper model.

The threads are the {user_threads:,} topics that ptrblck, the forum's most active answerer, took part in, plus {recent_threads:,} recent solved topics from other answerers for evaluation. The answer kept for each thread is whatever resolved it, whoever wrote it: ptrblck, another user, or the asker's own fix.

## Quick start

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
| `sft` | validation | {validation:,} | Same format, threads created from {valid_since} to {eval_since} |
| `eval` | test | {eval:,} | Questions with a reference answer, threads created on or after {eval_since} |
| `threads` | train | {threads:,} | Every crawled thread, cleaned, with its resolution and why it was excluded, if it was |

## How answers were chosen

1. **Accepted solution.** If a post is marked as the solution, it is the answer. When the asker marked their own thank-you post, the answer is the post they were thanking.
2. **Asker confirmation.** Otherwise, if the asker later said the fix worked ("that solved it", "works now"), the answer is the post they were replying to.
3. **Review.** Answers that point at other posts, are only a question, are under 60 characters, are mostly a link, or credit another post were reviewed by a Claude Haiku agent following [`prompts/review.md`](https://github.com/samadpls/torchlight/blob/main/prompts/review.md). The agent kept the post, dropped the thread, or composed one answer from several posts. Every evaluation answer was reviewed the same way. Composed answers may only use the source posts, and their code must appear verbatim in those posts, or they are dropped.

Threads with no accepted solution and no confirmation are kept in `threads` but produce no examples.

**Tiers:** `gold` answers are accepted solutions; `silver` answers were confirmed by the asker. The `eval` config is gold only.

## Example kinds in `sft`

- `answer`: the conversation up to the answer, ending with the answer as the final assistant turn. Earlier assistant turns are context only.
- `clarification`: the opening question and a first reply that asked for missing details, such as a minimal executable snippet or the full error. These teach the model to ask instead of guessing.

## Filters

{filters}

## Fields

- `sft`: `messages`, `kind`, `tier`, `answer_origin` (`verbatim` or `composed`), `answer_post_numbers`, `answer_authors`, `thread_id`, `url`, `created_at`
- `eval`: `question`, `reference_answer`, `needs_clarification`, `answered_by_ptrblck`, `tier`, `answer_origin`, `answer_post_numbers`, `answer_authors`, `thread_id`, `url`, `created_at`
- `threads`: `thread_id`, `url`, `title`, `created_at`, `category_id`, `tags`, `asker`, `split`, `resolution`, `excluded`, `posts`

Each row links to its source thread through `url`.

## Limitations

- Answers are as correct as the forum's resolution signals. An accepted or confirmed answer can still be imperfect or tied to an older PyTorch version.
- The data leans toward the questions ptrblck answered, so it is heavy on debugging, CUDA, training loops and shape errors.
- Images were replaced with `[image]`. Threads whose question or answer depends on an image were excluded.
- Emails and strings that look like API keys were redacted.

## License and attribution

Forum content is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by the [LF Projects Terms of Use](https://lfprojects.org/policies/terms-of-use/) that the PyTorch forums follow, and this dataset is released under the same license. Credit goes to the forum's authors; every row carries the URL of the thread it came from.

## Author

Built by Abdul Samad ([samadpls.com](https://samadpls.com)) with [torchlight](https://github.com/samadpls/torchlight). Questions and corrections: abdulsamadsid1@gmail.com.

## Citation

```bibtex
@misc{{samad2026torchlight,
  author = {{Abdul Samad}},
  title = {{torchlight threads: resolved PyTorch forum questions and answers}},
  year = {{2026}},
  url = {{https://huggingface.co/datasets/samadpls/torchlight-threads}}
}}
```
