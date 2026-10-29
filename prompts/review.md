# Review flagged PyTorch forum threads

You are reviewing threads from the PyTorch forums (discuss.pytorch.org). For each thread, the pipeline picked a candidate answer post automatically and flagged it because it couldn't be sure the post works as a standalone answer. Decide what the correct answer to the asker's question is, using only the posts shown.

## Input

The batch file has one section per thread:

```
=== thread <id> ===
Title: ...
Asker: <username>
Candidate answer: post <n> (<why it was flagged>)

--- post 1 by <username> ---
<content>
--- post 2 by <username> [accepted] ---
<content>
```

`[accepted]` marks the post the asker or a moderator accepted as the solution. Posts after the candidate are shown so you can see whether the asker confirmed the fix.

## Output

Write one block per thread, in the same order as the batch:

```
=== thread 123 ===
verdict: compose
sources: 2, 4
reason: fix is split between the suggestion and the follow-up
answer:
<the combined answer, as Markdown, until the next "=== thread" line>
```

- `verdict`
  - `keep`: the candidate post is a correct answer and makes sense on its own. Set `sources` to the candidate and leave out `answer:`.
  - `compose`: the fix is spread across several posts, or the candidate only makes sense together with another post. Write the combined answer after `answer:`, and list every post you used in `sources`.
  - `drop`: the posts don't contain a working resolution, the answer only exists in an image, or the thread isn't about PyTorch. Leave out `sources` and `answer:`.
- `reason`: a few words explaining the verdict.

## Rules for composed answers

- Use only what the posts say. Add no facts, no APIs and no code of your own.
- Copy code blocks exactly as they appear in the posts, keeping each block separate. Don't merge, edit or shorten code.
- Write it as a direct answer to the asker: no greetings, no "as X said", no forum usernames, no mention of posts or the thread.
- Keep it as short as the fix allows. Markdown is fine.

Be strict. If you aren't confident the posts contain a working fix, choose `drop`.
