import json
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

from torchlight.scrape.cache import ResponseCache
from torchlight.scrape.client import ForumClient
from torchlight.scrape.endpoints import (
    REPLIES_FILTER,
    SOLVED_FILTER,
    fetch_topics,
    recent_solved_topic_ids,
    user_action_topic_ids,
)


async def crawl(
    raw_dir: Path,
    username: str,
    eval_since: date,
    rate: float,
    concurrency: int,
    limit: int | None,
    log: Callable[[str], None],
) -> None:
    """Cache every topic the user posted in, plus recent solved topics for evaluation.

    Order of work: the user's solved topics, recent solved topics from anyone, then every
    other topic the user replied in. Writes `topic_ids.json` next to the cache.
    """
    cache = ResponseCache(raw_dir / "forum.sqlite")
    client = ForumClient(cache, rate)
    try:
        summary = (await client.get_json(f"/u/{username}/summary.json"))["user_summary"]
        solved = await user_action_topic_ids(
            client, username, SOLVED_FILTER, summary["solved_count"], concurrency
        )
        log(f"{username} solved topics: {len(solved)}")
        recent = await recent_solved_topic_ids(client, eval_since)
        log(f"solved topics since {eval_since}: {len(recent)}")
        replied = await user_action_topic_ids(
            client, username, REPLIES_FILTER, summary["post_count"], concurrency
        )
        log(f"{username} replied topics: {len(replied)}")

        (raw_dir / "topic_ids.json").write_text(
            json.dumps(
                {
                    "username": username,
                    "eval_since": eval_since.isoformat(),
                    "solved": solved,
                    "replied": replied,
                    "recent_solved": recent,
                }
            )
        )

        ordered = list(dict.fromkeys(solved + recent + replied))[:limit]
        started = time.monotonic()

        def progress(done: int) -> None:
            if done % 500 == 0 or done == len(ordered):
                minutes = (time.monotonic() - started) / 60
                log(
                    f"topics {done}/{len(ordered)}, requests {client.requests_made}, "
                    f"{minutes:.1f} min"
                )

        await fetch_topics(client, ordered, concurrency, progress)
    finally:
        await client.aclose()
        cache.close()
