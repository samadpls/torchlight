# Evaluating the model

This folder tests the fine-tuned model against models that were not fine-tuned but can search the forum instead. Four setups answer the same 260 questions. Then Claude Opus 5.5, acting as judge, scores each answer from 1 to 5 by comparing it with the answer that solved the thread.

## Results

| Model | Average grade (out of 5) | Graded 4 or 5 | Wrong answers (graded 1) | Code with only real torch APIs | Asked for details, when they were missing | Asked for details, when not needed | Searches per question |
|---|---|---|---|---|---|---|---|
| Phi-4-mini (base model) | 1.88 | 5% | 39% | 170 of 175 | 0 of 7 | 0% | 0.0 |
| Phi-4-mini + forum search | 1.75 | 5% | 45% | 145 of 157 | 1 of 7 | 2% | 0.1 |
| torchlight (fine-tuned) | 1.97 | 7% | 27% | 9 of 10 | 0 of 7 | 7% | 0.0 |
| torchlight + forum search | 1.95 | 6% | 26% | 4 of 4 | 0 of 7 | 2% | 0.0 |

**What the results show**

- **Far fewer wrong answers.** 27% of torchlight's answers were graded 1 (wrong), against 39% for the base model: 71 against 102. That gap is real, not chance (p = 0.003).
- **Better than giving the base model forum search.** torchlight averages 1.97 against 1.75 (95% interval +0.09 to +0.35). Search pulled the small base model off topic: on the 23 questions where it searched, it averaged 1.35.
- **Short and fast.** The median answer is 31 tokens, against 690 for the base model, and it answered all 260 questions in 9 minutes, against 32.
- **The test is hard by design.** The questions are real forum posts from 2025 and 2026, after the model's knowledge ends, and a 4 or 5 needs the same fix that solved the thread. Every small setup gets a 4 or 5 on only 5–7% of questions. torchlight raises the average grade from 1.88 to 1.97 and good answers from 14 to 19, but those gaps are too small to be sure of.
- **torchlight never uses the search tool.** None of its training chats included tools, so adding search changes nothing.
- **A bigger or newer base model would likely do better.** Fine-tuning taught the model the forum's style, but what it knows is limited by its size (3.8B parameters) and by data that ends in mid-2024. The same training on a larger or more recent model is the next step. We haven't tested it yet.

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
