import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dataset_card.md")


def dataset_card(stats_path: Path, valid_since: str, eval_since: str) -> str:
    stats = json.loads(stats_path.read_text())
    filters = "\n".join(
        f"- {key.removeprefix('excluded: ').capitalize()}: {value:,} threads"
        for key, value in stats.items()
        if key.startswith("excluded: ")
    )
    return TEMPLATE.read_text().format(
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
