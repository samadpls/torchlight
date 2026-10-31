import random
import re
from pathlib import Path

from pydantic import BaseModel, Field

from torchlight.eval.generate import Answer
from torchlight.schemas import EvalExample

ITEM_HEADER = re.compile(r"^=== item (\d+) ===[ \t]*$", re.MULTILINE)
GRADE_LINE = re.compile(r"candidate (\d+): score ([1-5]) \| asks (yes|no)", re.IGNORECASE)


class Grade(BaseModel):
    thread_id: int
    arm: str
    score: int = Field(ge=1, le=5)
    asks: bool


def write_batches(blocks: list[str], batch_dir: Path, batch_size: int) -> int:
    batch_dir.mkdir(parents=True, exist_ok=True)
    for old in batch_dir.glob("batch_*.md"):
        old.unlink()
    starts = range(0, len(blocks), batch_size)
    for index, start in enumerate(starts):
        chunk = blocks[start : start + batch_size]
        (batch_dir / f"batch_{index:04d}.md").write_text("\n\n".join(chunk) + "\n")
    return len(starts)


def _blocks(path: Path) -> list[tuple[int, str]]:
    parts = ITEM_HEADER.split(path.read_text())
    return [(int(i), body.strip()) for i, body in zip(parts[1::2], parts[2::2], strict=True)]


def question_block(example: EvalExample) -> str:
    return f"=== item {example.thread_id} ===\n{example.question}"


def load_answers(results_dir: Path, arm: str) -> list[Answer]:
    """Answers written by agents in the prompts/answer.md format."""
    answers = []
    for path in sorted(results_dir.glob("*.md")):
        for thread_id, body in _blocks(path):
            head, _, answer = body.partition("answer:")
            searches = []
            for line in head.splitlines():
                if line.lower().startswith("searches:"):
                    searches = [q.strip() for q in line.split(":", 1)[1].split("|") if q.strip()]
            answers.append(
                Answer(
                    arm=arm,
                    thread_id=thread_id,
                    answer=answer.strip(),
                    searches=searches,
                    generated_tokens=0,
                )
            )
    return answers


def judge_order(thread_id: int, arms: list[str]) -> list[str]:
    """The arms in the order their answers are shown, shuffled per item but reproducibly."""
    order = sorted(arms)
    random.Random(thread_id).shuffle(order)
    return order


def judge_block(example: EvalExample, answers: dict[str, Answer]) -> str:
    lines = [
        f"=== item {example.thread_id} ===",
        "Question:",
        example.question,
        "",
        "Reference answer:",
        example.reference_answer,
    ]
    for n, arm in enumerate(judge_order(example.thread_id, list(answers)), start=1):
        lines += ["", f"--- candidate {n} ---", answers[arm].answer or "(no answer)"]
    return "\n".join(lines)


def load_grades(results_dir: Path, arms_by_item: dict[int, list[str]]) -> tuple[list[Grade], int]:
    """Grades mapped back to arms, and how many items were missing or malformed."""
    grades = []
    invalid = 0
    for path in sorted(results_dir.glob("*.md")):
        for thread_id, body in _blocks(path):
            order = judge_order(thread_id, arms_by_item.get(thread_id, []))
            found = {int(n): (int(s), a.lower() == "yes") for n, s, a in GRADE_LINE.findall(body)}
            if not order or sorted(found) != list(range(1, len(order) + 1)):
                invalid += 1
                continue
            grades += [
                Grade(thread_id=thread_id, arm=order[n - 1], score=score, asks=asks)
                for n, (score, asks) in found.items()
            ]
    return grades, invalid
