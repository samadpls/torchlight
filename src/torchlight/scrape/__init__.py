"""Everything that talks to the PyTorch forums.

- `client`: rate-limited HTTP client with retries
- `cache`: SQLite cache of raw responses, so every later stage runs offline
- `endpoints`: the Discourse endpoints we read, and how long threads are paged
- `crawl`: the crawl itself
"""

from torchlight.scrape.cache import ResponseCache
from torchlight.scrape.client import FORUM_URL, USER_AGENT, ForumClient
from torchlight.scrape.crawl import crawl
from torchlight.scrape.endpoints import remaining_posts_paths, topic_path

__all__ = [
    "FORUM_URL",
    "USER_AGENT",
    "ForumClient",
    "ResponseCache",
    "crawl",
    "remaining_posts_paths",
    "topic_path",
]
