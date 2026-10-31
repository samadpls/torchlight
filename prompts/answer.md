# Answer PyTorch forum questions with a forum search tool

You are one of the assistants being compared on questions from the PyTorch forums. Answer each question as this assistant:

> {system_prompt}

You can search solved threads from the PyTorch forums with

```
uv run torchlight search "<query>"
```

which prints the closest questions with the answers that resolved them. Search at most twice per question, only when it would help, and use what you find only if it fits the question. Don't use any other tool, file or website.

## Input

```
=== item <id> ===
<question>
```

## Output

One block per item, in the same order:

```
=== item 123 ===
searches: <first query> | <second query>
answer:
<your answer, as Markdown, until the next "=== item" line>
```

Leave `searches:` empty when you didn't search. Write the answer to the asker directly: no mention of the search tool, the forum threads you read, or this task.
