import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dataset_card.md")

FILTER_LABELS = {
    "excluded: unresolved": "No accepted answer and no confirmation from the asker",
    "excluded: created before PyTorch 0.4": "Created before PyTorch 0.4 (May 2018)",
    "excluded: question depends on an image": "Question depends on an image",
    "excluded: answer is an image": "Answer is an image",
    "excluded: too long": "Longer than 8 messages or 12,000 characters",
    "excluded: outdated API in answer": "Answer uses removed APIs such as `volatile=True`",
    "excluded: not English": "Not in English",
    "excluded: duplicate question": "Duplicate question",
    "dropped in review": "Dropped by the review agent",
    "composed answer not grounded": "Combined answer whose code was not found in its sources",
    "eval: silver skipped": "Evaluation threads without an accepted answer",
}


def _count(value: int) -> str:
    return f"{value:,} thread" + ("" if value == 1 else "s")


def dataset_card(stats_path: Path, valid_since: str, eval_since: str) -> str:
    stats = json.loads(stats_path.read_text())
    filters = "\n".join(
        f"- {label}: {_count(stats[key])}" for key, label in FILTER_LABELS.items() if stats.get(key)
    )
    splits = ("train", "validation")
    gold = sum(stats.get(f"examples: {s} answer gold", 0) for s in splits)
    silver = sum(stats.get(f"examples: {s} answer silver", 0) for s in splits)
    clarifications = sum(
        stats.get(f"examples: {s} clarification {t}", 0) for s in splits for t in ("gold", "silver")
    )
    usable = sum(stats.get(f"examples: {s}", 0) for s in (*splits, "eval"))
    review = {v: stats.get(f"review: {v}", 0) for v in ("keep", "compose")}
    review_drop = stats.get("dropped in review", 0)
    return TEMPLATE.read_text().format(
        usable=usable,
        usable_badge=f"{usable:,}".replace(",", "%2C"),
        usable_threads=gold + silver + stats.get("examples: eval", 0),
        answers=gold + silver,
        gold_answers=gold,
        silver_answers=silver,
        clarifications=clarifications,
        reviewed=review["keep"] + review["compose"] + review_drop,
        review_keep=review["keep"],
        review_compose=review["compose"],
        review_drop=review_drop,
        threads=sum(v for k, v in stats.items() if k.startswith("threads: ")),
        user_threads=stats.get("source: user topics", 0),
        recent_threads=stats.get("source: other recent solved topics", 0),
        train=stats.get("examples: train", 0),
        validation=stats.get("examples: validation", 0),
        eval=stats.get("examples: eval", 0),
        valid_since=valid_since,
        eval_since=eval_since,
        filters=filters,
    )
