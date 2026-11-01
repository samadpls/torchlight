# Evaluating the models

This folder tests the fine-tuned models against models that were not fine-tuned but can search the forum instead. Four setups answer the same 260 questions. Then Claude Opus 5.5, acting as judge, scores each answer from 1 to 5 by comparing it with the answer that solved the thread.

## Results

| Model | Average grade (out of 5) | Graded 4 or 5 | Wrong answers (graded 1) | Code with only real torch APIs | Asked for details, when they were missing | Asked for details, when not needed | Searches per question |
|---|---|---|---|---|---|---|---|
| Phi-4-mini (base model) | 1.88 | 5% | 39% | 170 of 175 | 0 of 7 | 0% | 0.0 |
| Phi-4-mini + forum search | 1.75 | 5% | 45% | 145 of 157 | 1 of 7 | 2% | 0.1 |
| torchlight-phi4-mini (fine-tuned) | 1.97 | 7% | 27% | 9 of 10 | 0 of 7 | 7% | 0.0 |
| torchlight-phi4-mini + forum search | 1.95 | 6% | 26% | 4 of 4 | 0 of 7 | 2% | 0.0 |

**What the results show**

- **Far fewer wrong answers.** 27% of torchlight-phi4-mini's answers were graded 1 (wrong), against 39% for the base model: 71 against 102. That gap is real, not chance (p = 0.003).
- **Better than giving the base model forum search.** torchlight-phi4-mini averages 1.97 against 1.75 (95% interval +0.09 to +0.35). Search pulled the small base model off topic: on the 23 questions where it searched, it averaged 1.35.
- **Short and fast.** The median answer is 31 tokens, against 690 for the base model, and it answered all 260 questions in 9 minutes, against 32.
- **The test is hard by design.** The questions are real forum posts from 2025 and 2026, after the model's knowledge ends, and a 4 or 5 needs the same fix that solved the thread. Every small setup gets a 4 or 5 on only 5–7% of questions. torchlight-phi4-mini raises the average grade from 1.88 to 1.97 and good answers from 14 to 19, but those gaps are too small to be sure of.
- **torchlight-phi4-mini never uses the search tool.** None of its training chats included tools, so adding search changes nothing.
- **A bigger model helps.** The same training on Qwen3-8B (8.2B parameters) scored higher on the newest questions. See [`torchlight-qwen3-8b`](https://huggingface.co/samadpls/torchlight-qwen3-8b).

## Second comparison: Qwen3-8B

The second model, `torchlight-qwen3-8b`, was trained and run on Kaggle (see [`kaggle/`](../../../kaggle/)). Qwen3-8B's training data reportedly runs to March 2025, so this comparison uses only the 163 questions asked from April 2025 on. Four models answer them without forum search: Phi-4-mini and Qwen3-8B, each before and after fine-tuning. The judge grades all four together.

| Model | Average grade (out of 5) | Graded 4 or 5 | Wrong answers (graded 1) | Code with only real torch APIs | Asked for details, when they were missing | Asked for details, when not needed |
|---|---|---|---|---|---|---|
| Phi-4-mini (base model) | 1.96 | 5% | 33% | 108 of 109 | 0 of 4 | 1% |
| torchlight-phi4-mini (fine-tuned) | 2.09 | 8% | 16% | 5 of 5 | 0 of 4 | 5% |
| Qwen3-8B (base model) | 2.17 | 17% | 31% | 114 of 120 | 0 of 4 | 1% |
| torchlight-qwen3-8b (fine-tuned) | 2.25 | 9% | 15% | 11 of 11 | 3 of 4 | 13% |

**What the results show**

- **Half as many wrong answers.** Fine-tuning cut answers graded 1 (wrong) from 31% to 15% for Qwen3-8B, and from 33% to 16% for Phi-4-mini. Both gaps are real, not chance (p < 0.001).
- **The bigger model helps.** Qwen3-8B beats Phi-4-mini both before fine-tuning (2.17 against 1.96) and after (2.25 against 2.09). Both gaps are real (95% intervals +0.04 to +0.39 and +0.03 to +0.29).
- **It asks for missing details.** On the 4 questions where the expert first asked for more information, torchlight-qwen3-8b asked too on 3. None of the other models asked on any. It also asks when it doesn't need to on 13% of the other questions.
- **But it gives fewer complete fixes than Qwen3-8B itself.** 9% of its answers were graded 4 or 5, against 17% for the base model (p = 0.02). The base model writes long answers (a median of 1,024 tokens, often cut off) that more often contain the exact fix. The fine-tune learned the forum's short style (a median of 42 tokens).
- **On average, the fine-tune is a little ahead** of Qwen3-8B (2.25 against 2.17), but that gap is too small to be sure of (95% interval −0.09 to +0.23).
- **The test is hard by design.** These 163 questions are newer than Qwen3-8B's training data, and a 4 or 5 needs the same fix that solved the thread.

```bash
uv run torchlight batches judge --eval-set qwen --since 2025-04-01
uv run torchlight report --eval-set qwen --since 2025-04-01
```

The answers live in `data/eval/qwen/answers/`: arms A and D copied from the first comparison, and Q and QF from the Kaggle jobs.

## The four setups

| Setup | Can search the forum | Letter in `--arm` |
|---|---|---|
| Phi-4-mini (the base model) | no | A |
| Phi-4-mini + forum search | yes | B |
| torchlight (fine-tuned) | no | D |
| torchlight + forum search | yes | E |

All four use the same model size: with and without fine-tuning, and with and without search.

- **Questions.** The 260 questions in the dataset's `eval` config. They were asked between 2025-01-01 and 2026-10-08, after the base model's data ends (June 2024) and after every training thread.
- **Search.** The `search_forum` tool does keyword search (BM25) over 7,438 solved training threads and returns the top 3. A model may search up to twice per question. It decides itself whether to search.
- **Prompt.** Every setup gets the same system prompt as in training.
- **Settings.** The local models use greedy decoding (no sampling) and up to 1,024 new tokens.
- **Reference answer.** Each question also got an answer from Claude Sonnet 5.5 with the same search tool, written by Claude Code agents from `torchlight batches questions`. The judge scored it too, but it isn't in the results, which compare models of the same size.

## How answers are scored

The judge is Claude Opus 5.5, run as Claude Code agents that follow [`prompts/judge.md`](../../../prompts/judge.md). For each question it sees the question, the answer that solved the thread, and all the answers, shuffled and without names. For each answer it gives:

- **A score from 1 to 5.** 5 means the same fix as the real answer, or one just as correct. 3 means on topic but too general. 1 means wrong, or uses functions that PyTorch doesn't have.
- **Whether it asks for more details.** Some questions can't be answered without the user's code or full error. For those, asking is the right move, and scores well.

A script also checks the code in each answer: every `torch.*` name must exist in PyTorch 2.15. This check doesn't use the judge.

## Run it

```bash
uv run torchlight index               # build the search index
uv run torchlight answer --arm A      # also B, D and E; 10 to 35 minutes each on a MacBook Pro (M3 Pro)
uv run torchlight batches questions   # questions for the reference answers
uv run torchlight batches judge       # scoring batches, once every setup has answered
uv run torchlight report              # the results table
```

The agents write Claude's answers to `data/eval/agent/results/` and the scores to `data/eval/judge/results/`. `report` saves the table to `data/eval/results.md`.

## Files

| File | What it does |
|---|---|
| [`retrieval/search.py`](../retrieval/search.py) | The `search_forum` tool |
| `generate.py` | Runs the local models, handles their tool calls, and groups questions by length so batches stay fast |
| `batches.py` | Writes the batches for Claude and the judge, and reads their results back |
| `apis.py` | Checks that the torch names in an answer exist |
| `report.py` | Builds the results table |
| [`prompts/answer.md`](../../../prompts/answer.md), [`prompts/judge.md`](../../../prompts/judge.md) | Instructions for Claude and the judge |
