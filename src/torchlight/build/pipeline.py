import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from torchlight.build.clean import depends_on_image, is_mostly_non_latin, uses_outdated_api
from torchlight.build.examples import (
    answer_messages,
    clarification_messages,
    first_helper_reply,
    needs_clarification,
    question_text,
)
from torchlight.build.parse import load_thread
from torchlight.build.resolve import resolve
from torchlight.build.review import code_is_grounded, load_results, review_item, write_batches
from torchlight.schemas import EvalExample, Resolution, SFTExample, Thread
from torchlight.scrape import ResponseCache

PYTORCH_04_RELEASE = date(2018, 5, 1)
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _exclusion(thread: Thread, resolution: Resolution) -> str | None:
    question = thread.posts[0].content
    if thread.created_at.date() < PYTORCH_04_RELEASE:
        return "created before PyTorch 0.4"
    if resolution.status == "unresolved":
        return "unresolved"
    if is_mostly_non_latin(question):
        return "not English"
    if depends_on_image(question):
        return "question depends on an image"
    if depends_on_image(thread.post(resolution.answer_post_number).content):
        return "answer is an image"
    return None


def _thread_record(
    thread: Thread, resolution: Resolution, split: str, excluded: str | None
) -> dict:
    return {
        "thread_id": thread.id,
        "url": thread.url,
        "title": thread.title,
        "created_at": thread.created_at.isoformat(),
        "category_id": thread.category_id,
        "tags": thread.tags,
        "asker": thread.asker,
        "split": split,
        "resolution": resolution.model_dump(),
        "excluded": excluded,
        "posts": [p.model_dump(mode="json", exclude={"id"}) for p in thread.posts],
    }


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write((row if isinstance(row, str) else row.model_dump_json()) + "\n")


def _question_key(thread: Thread) -> str:
    return NON_ALNUM.sub(" ", question_text(thread).lower()).strip()[:400]


def build(
    data_dir: Path, valid_since: date, batch_size: int, review_eval: bool
) -> tuple[Counter, int]:
    """Build every dataset file from the crawl cache; returns stats and the pending batch count."""
    raw_dir, review_dir, out_dir = data_dir / "raw", data_dir / "review", data_dir / "processed"
    ids = json.loads((raw_dir / "topic_ids.json").read_text())
    eval_since = date.fromisoformat(ids["eval_since"])
    username = ids["username"]
    topic_ids = list(dict.fromkeys(ids["solved"] + ids["recent_solved"] + ids["replied"]))

    cache = ResponseCache(raw_dir / "forum.sqlite")
    results, invalid = load_results(review_dir / "results")
    ptrblck_topics = set(ids["solved"]) | set(ids["replied"])
    stats: Counter = Counter(
        {
            "invalid review blocks": invalid,
            "source: user topics": len(ptrblck_topics),
            "source: other recent solved topics": len(set(ids["recent_solved"]) - ptrblck_topics),
        }
    )
    threads: list[str] = []
    splits: dict[str, list] = {"train": [], "validation": [], "eval": []}
    pending: list[str] = []
    seen_questions: set[str] = set()

    for topic_id in topic_ids:
        thread = load_thread(cache, topic_id)
        if thread is None:
            stats["missing from cache"] += 1
            continue
        created = thread.created_at.date()
        split = "eval" if created >= eval_since else "train"
        if split == "train" and created >= valid_since:
            split = "validation"
        resolution = resolve(thread)
        excluded = _exclusion(thread, resolution)
        threads.append(json.dumps(_thread_record(thread, resolution, split, excluded)))
        stats[f"threads: {split}"] += 1
        if excluded:
            stats[f"excluded: {excluded}"] += 1
            continue

        answer_post = thread.post(resolution.answer_post_number)
        answer, origin, sources = answer_post.content, "verbatim", [answer_post.post_number]
        reasons = list(resolution.review_reasons)
        if split == "eval" and review_eval:
            reasons.append("evaluation answer")
        result = results.get(thread.id)
        if result is None and reasons:
            pending.append(review_item(thread, resolution, reasons))
            stats["pending review"] += 1
            continue
        if result is not None:
            if result.verdict == "drop":
                stats["dropped in review"] += 1
                continue
            if result.verdict == "compose":
                source_posts = [thread.post(n) for n in result.source_post_numbers]
                if None in source_posts or not code_is_grounded(
                    result.answer, [p.content for p in source_posts]
                ):
                    stats["composed answer not grounded"] += 1
                    continue
                answer, origin, sources = result.answer, "composed", result.source_post_numbers
            stats[f"review: {result.verdict}"] += 1

        if uses_outdated_api(answer):
            stats["excluded: outdated API in answer"] += 1
            continue
        key = _question_key(thread)
        if key in seen_questions:
            stats["excluded: duplicate question"] += 1
            continue
        seen_questions.add(key)

        tier = "gold" if resolution.status == "accepted" else "silver"
        authors = sorted({thread.post(n).username for n in sources})
        common = {
            "thread_id": thread.id,
            "url": thread.url,
            "created_at": thread.created_at,
            "tier": tier,
        }
        if split == "eval":
            if tier != "gold":
                stats["eval: silver skipped"] += 1
                continue
            splits["eval"].append(
                EvalExample(
                    question=question_text(thread),
                    reference_answer=answer,
                    needs_clarification=needs_clarification(thread, answer_post.post_number),
                    answer_origin=origin,
                    answer_post_numbers=sources,
                    answer_authors=authors,
                    answered_by_ptrblck=username in authors,
                    **common,
                )
            )
            continue

        single_turn = origin == "composed" or answer_post.username == thread.asker
        messages = answer_messages(thread, answer_post.post_number, answer, single_turn)
        if messages is None:
            stats["excluded: too long"] += 1
            continue
        splits[split].append(
            SFTExample(
                messages=messages,
                kind="answer",
                answer_origin=origin,
                answer_post_numbers=sources,
                answer_authors=authors,
                **common,
            )
        )
        clarification = clarification_messages(thread, answer_post.post_number)
        if clarification is not None:
            reply = first_helper_reply(thread)
            splits[split].append(
                SFTExample(
                    messages=clarification,
                    kind="clarification",
                    answer_origin="verbatim",
                    answer_post_numbers=[reply.post_number],
                    answer_authors=[reply.username],
                    **common,
                )
            )
    cache.close()

    for rows in splits.values():
        rows.sort(key=lambda r: (r.created_at, r.thread_id))
    _write_jsonl(out_dir / "threads" / "threads.jsonl", threads)
    _write_jsonl(out_dir / "sft" / "train.jsonl", splits["train"])
    _write_jsonl(out_dir / "sft" / "validation.jsonl", splits["validation"])
    _write_jsonl(out_dir / "eval" / "test.jsonl", splits["eval"])
    for name, rows in splits.items():
        stats[f"examples: {name}"] = len(rows)
        for row in rows:
            stats[f"examples: {name} {getattr(row, 'kind', 'eval')} {row.tier}"] += 1
            stats[f"examples: {name} {row.answer_origin}"] += 1
    batches = write_batches(pending, review_dir / "batches", batch_size)
    (out_dir / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2))
    return stats, len(batches)
