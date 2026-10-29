import re
from pathlib import Path

from pydantic import ValidationError

from torchlight.schemas import Resolution, ReviewResult, Thread

MAX_POST_CHARS = 2_000
POSTS_AFTER_ANSWER = 2
CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
WHITESPACE = re.compile(r"\s+")
RESULT_HEADER = re.compile(r"^=== thread (\d+) ===[ \t]*$", re.MULTILINE)


def review_item(thread: Thread, resolution: Resolution, reasons: list[str]) -> str:
    answer = resolution.answer_post_number
    shown = [p for p in thread.posts if p.post_number <= answer + POSTS_AFTER_ANSWER]
    lines = [
        f"=== thread {thread.id} ===",
        f"Title: {thread.title}",
        f"Asker: {thread.asker}",
        f"Candidate answer: post {answer} ({'; '.join(reasons)})",
    ]
    for post in shown:
        content = post.content
        if len(content) > MAX_POST_CHARS:
            content = content[:MAX_POST_CHARS] + "\n[truncated]"
        marks = " [accepted]" if post.accepted_answer else ""
        lines += ["", f"--- post {post.post_number} by {post.username}{marks} ---", content]
    return "\n".join(lines)


def write_batches(items: list[str], batch_dir: Path, batch_size: int) -> list[Path]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    for old in batch_dir.glob("batch_*.md"):
        old.unlink()
    paths = []
    for index, start in enumerate(range(0, len(items), batch_size)):
        path = batch_dir / f"batch_{index:04d}.md"
        path.write_text("\n\n".join(items[start : start + batch_size]) + "\n")
        paths.append(path)
    return paths


def parse_result_block(thread_id: int, body: str) -> ReviewResult:
    fields: dict = {"thread_id": thread_id}
    head, _, answer = body.partition("\nanswer:")
    for line in head.splitlines():
        key, sep, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if not sep:
            continue
        if key == "verdict":
            fields["verdict"] = value.lower()
        elif key == "sources":
            fields["source_post_numbers"] = [int(n) for n in re.findall(r"\d+", value)]
        elif key == "reason":
            fields["reason"] = value
    if fields.get("verdict") == "compose":
        fields["answer"] = answer.strip() or None
    return ReviewResult.model_validate(fields)


def load_results(results_dir: Path) -> tuple[dict[int, ReviewResult], int]:
    """Valid results keyed by thread id, and how many blocks failed validation."""
    results: dict[int, ReviewResult] = {}
    invalid = 0
    for path in sorted(results_dir.glob("*.md")):
        parts = RESULT_HEADER.split(path.read_text())
        for thread_id, body in zip(parts[1::2], parts[2::2], strict=True):
            try:
                result = parse_result_block(int(thread_id), "\n" + body.strip())
            except (ValueError, ValidationError):
                invalid += 1
                continue
            if result.verdict == "compose" and not (result.answer and result.source_post_numbers):
                invalid += 1
                continue
            results[result.thread_id] = result
    return results, invalid


def _normalize(code: str) -> str:
    return WHITESPACE.sub("", code)


def code_is_grounded(answer: str, sources: list[str]) -> bool:
    """Every code block in a composed answer must appear in one of its source posts."""
    source_code = [_normalize(c) for s in sources for c in CODE_BLOCK.findall(s)]
    source_text = _normalize("\n".join(sources))
    for block in CODE_BLOCK.findall(answer):
        code = _normalize(block)
        if code and not any(code in s for s in source_code) and code not in source_text:
            return False
    return True
