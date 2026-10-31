# Evaluating the model

This folder tests the fine-tuned model against models that were not fine-tuned but can search the forum instead. Five setups answer the same 260 questions. Then Claude Opus 5.5, acting as judge, scores each answer from 1 to 5 by comparing it with the answer that solved the thread.

## Results

The evaluation is still running. Results will be added here when it finishes.

## The five setups

| Setup | Can search the forum | Letter in `--arm` |
|---|---|---|
| Phi-4-mini (the base model) | no | A |
| Phi-4-mini + forum search | yes | B |
| torchlight (fine-tuned) | no | D |
| torchlight + forum search | yes | E |
| Claude Sonnet 5.5 + forum search (reference) | yes | C |

The four Phi-4-mini setups are the real comparison: same size, with and without fine-tuning, with and without search. Claude Sonnet 5.5 shows the ceiling.

- **Questions.** The 260 questions in the dataset's `eval` config. They were asked between 2025-01-01 and 2026-10-08, after the base model's data ends (June 2024) and after every training thread.
- **Search.** The `search_forum` tool does keyword search (BM25) over 7,438 solved training threads and returns the top 3. A model may search up to twice per question. It decides itself whether to search.
- **Prompt.** Every setup gets the same system prompt as in training.
- **Settings.** The local models use greedy decoding (no sampling) and up to 1,024 new tokens.
- **Claude Sonnet 5.5.** Runs as Claude Code agents with the same search tool. It has probably seen many of these threads, so its row is a reference point.

## How answers are scored

The judge is Claude Opus 5.5, run as Claude Code agents that follow [`prompts/judge.md`](../../../prompts/judge.md). For each question it sees the question, the answer that solved the thread, and the five answers, shuffled and without names. For each answer it gives:

- **A score from 1 to 5.** 5 means the same fix as the real answer, or one just as correct. 3 means on topic but too general. 1 means wrong, or uses functions that PyTorch doesn't have.
- **Whether it asks for more details.** Some questions can't be answered without the user's code or full error. For those, asking is the right move, and scores well.

A script also checks the code in each answer: every `torch.*` name must exist in PyTorch 2.15. This check doesn't use the judge.

## Run it

```bash
uv run torchlight index               # build the search index
uv run torchlight answer --arm A      # also B, D and E; 10 to 30 minutes each on a MacBook Pro (M3 Pro)
uv run torchlight batches questions   # questions for Claude Sonnet 5.5
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
