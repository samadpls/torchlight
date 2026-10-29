import asyncio
from collections.abc import Awaitable, Callable, Iterable
from datetime import date

from torchlight.scrape.client import ForumClient

PAGE_SIZE = 30
POSTS_PER_REQUEST = 20
REPLIES_FILTER = 5
SOLVED_FILTER = 15


def topic_path(topic_id: int) -> str:
    return f"/t/{topic_id}.json?include_raw=1"


def remaining_posts_paths(topic: dict) -> list[str]:
    """Paths for the posts that /t/{id}.json leaves out of long threads."""
    loaded = {post["id"] for post in topic["post_stream"]["posts"]}
    missing = [post_id for post_id in topic["post_stream"]["stream"] if post_id not in loaded]
    paths = []
    for start in range(0, len(missing), POSTS_PER_REQUEST):
        ids = "&".join(f"post_ids[]={i}" for i in missing[start : start + POSTS_PER_REQUEST])
        paths.append(f"/t/{topic['id']}/posts.json?{ids}&include_raw=1")
    return paths


async def gather_limited(coros: Iterable[Awaitable], concurrency: int) -> list:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(coro: Awaitable):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(run(c) for c in coros))


async def user_action_topic_ids(
    client: ForumClient, username: str, action_filter: int, expected: int, concurrency: int
) -> list[int]:
    offsets = range(0, expected + 3 * PAGE_SIZE, PAGE_SIZE)
    pages = await gather_limited(
        (
            client.get_json(
                f"/user_actions.json?offset={o}&username={username}&filter={action_filter}"
            )
            for o in offsets
        ),
        concurrency,
    )
    topic_ids: dict[int, None] = {}
    for page in pages:
        for action in (page or {}).get("user_actions", []):
            topic_ids.setdefault(action["topic_id"], None)
    return list(topic_ids)


async def recent_solved_topic_ids(client: ForumClient, since: date) -> list[int]:
    """Solved topics created on or after `since`, from any answerer, newest first."""
    topic_ids = []
    page = 0
    while True:
        data = await client.get_json(f"/latest.json?order=created&ascending=false&page={page}")
        topics = (data or {}).get("topic_list", {}).get("topics", [])
        if not topics:
            break
        for topic in topics:
            if topic["created_at"][:10] >= since.isoformat() and topic.get("has_accepted_answer"):
                topic_ids.append(topic["id"])
        if min(topic["created_at"][:10] for topic in topics) < since.isoformat():
            break
        page += 1
    return topic_ids


async def fetch_topics(
    client: ForumClient,
    topic_ids: list[int],
    concurrency: int,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    done = 0

    async def fetch(topic_id: int) -> None:
        nonlocal done
        topic = await client.get_json(topic_path(topic_id))
        if topic:
            for path in remaining_posts_paths(topic):
                await client.get_json(path)
        done += 1
        if on_progress:
            on_progress(done)

    await gather_limited((fetch(t) for t in topic_ids), concurrency)
