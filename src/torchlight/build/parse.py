from datetime import datetime

from torchlight.build.clean import clean_markdown
from torchlight.schemas import Post, Thread
from torchlight.scrape import ResponseCache, remaining_posts_paths, topic_path

REGULAR_POST = 1
LIKE_ACTION = 2



def _post(data: dict) -> Post | None:
    if (
        data.get("post_type") != REGULAR_POST
        or data.get("hidden")
        or data.get("deleted_at")
        or data.get("user_deleted")
    ):
        return None
    content = clean_markdown(data.get("raw") or "")
    if not content:
        return None
    likes = next(
        (a.get("count", 0) for a in data.get("actions_summary", []) if a.get("id") == LIKE_ACTION),
        0,
    )
    return Post(
        id=data["id"],
        post_number=data["post_number"],
        username=data["username"],
        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        content=content,
        reply_to_post_number=data.get("reply_to_post_number"),
        accepted_answer=bool(data.get("accepted_answer")),
        like_count=likes,
    )


def _tag_names(tags: list) -> list[str]:
    return [t["name"] if isinstance(t, dict) else str(t) for t in tags or []]


def load_thread(cache: ResponseCache, topic_id: int) -> Thread | None:
    topic = cache.get_json(topic_path(topic_id))
    if topic is None:
        return None
    raw_posts = list(topic["post_stream"]["posts"])
    for path in remaining_posts_paths(topic):
        extra = cache.get_json(path)
        if extra is None:
            return None
        raw_posts.extend(extra["post_stream"]["posts"])
    posts = sorted(
        (p for p in map(_post, raw_posts) if p is not None), key=lambda p: p.post_number
    )
    if not posts or posts[0].post_number != 1:
        return None
    return Thread(
        id=topic["id"],
        title=topic["title"],
        slug=topic["slug"],
        created_at=datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00")),
        category_id=topic.get("category_id"),
        tags=_tag_names(topic.get("tags")),
        posts=posts,
    )
