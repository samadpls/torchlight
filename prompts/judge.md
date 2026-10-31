# Grade answers to PyTorch forum questions

Each item is a question from the PyTorch forums (discuss.pytorch.org), the reference answer that resolved it on the forum, and several candidate answers written by different assistants. Grade every candidate against the reference.

## Input

```
=== item <id> ===
Question:
<question>

Reference answer:
<the answer that resolved the thread>

--- candidate 1 ---
<answer>
--- candidate 2 ---
<answer>
```

The candidates are in random order and don't say who wrote them. Judge each one on its own; don't compare their styles or lengths.

## Output

One block per item, in the same order, with one line per candidate:

```
=== item 123 ===
candidate 1: score 4 | asks no
candidate 2: score 2 | asks yes
```

### score (1 to 5): does the candidate solve the asker's problem the way the reference does?

- **5**: identifies the same cause and fix as the reference, or an equally correct one, with nothing wrong in it.
- **4**: correct and useful, but misses a detail the reference has, or is less specific.
- **3**: partly right: on topic and not wrong, but generic, incomplete, or the right fix is buried among guesses.
- **2**: mostly misses: the advice wouldn't fix the problem, or the main suggestion is wrong.
- **1**: wrong or harmful: incorrect claims, code that can't work, or functions and arguments PyTorch doesn't have.

When the reference asks the asker for more information (code, the full error, versions), the question couldn't be answered as asked. Then a candidate that asks for the same kind of missing information scores 4 or 5, and one that confidently guesses a fix scores at most 3 unless the guess matches what turned out to be the fix.

### asks (yes or no)

`yes` when the candidate asks the user for more information before or instead of answering, for example a code snippet, the full error message or their PyTorch version. A closing offer like "let me know if this doesn't work" is `no`.

Be strict and consistent: a long answer isn't better for being long, and confident wording isn't evidence of correctness.
