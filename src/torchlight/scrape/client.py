import asyncio
import json
import time

import httpx

from torchlight.scrape.cache import ResponseCache

FORUM_URL = "https://discuss.pytorch.org"
USER_AGENT = (
    "torchlight-dataset-builder (+https://github.com/samadpls/torchlight; abdulsamadsid1@gmail.com)"
)
RETRY_STATUSES = {429, 500, 502, 503, 504}


class RateLimiter:
    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self._interval
        if start > now:
            await asyncio.sleep(start - now)

    def pause(self, seconds: float) -> None:
        self._next = max(self._next, time.monotonic() + seconds)


def _retry_after(response: httpx.Response, attempt: int) -> float:
    try:
        return float(response.headers["Retry-After"])
    except (KeyError, ValueError):
        return 2.0 ** (attempt + 2)


class ForumClient:
    def __init__(
        self,
        cache: ResponseCache,
        rate: float,
        base_url: str = FORUM_URL,
        user_agent: str = USER_AGENT,
        max_retries: int = 6,
    ) -> None:
        self._cache = cache
        self._limiter = RateLimiter(rate)
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"User-Agent": user_agent},
            timeout=30.0,
            follow_redirects=True,
        )
        self.requests_made = 0

    async def get_json(self, path: str) -> dict | None:
        """Return the decoded body for a 200, None for other final statuses; cache both."""
        cached = self._cache.get(path)
        if cached is not None:
            status, body = cached
            return json.loads(body) if status == 200 else None

        for attempt in range(self._max_retries):
            await self._limiter.wait()
            self.requests_made += 1
            try:
                response = await self._http.get(path)
            except httpx.TransportError:
                await asyncio.sleep(2.0**attempt)
                continue
            if response.status_code in RETRY_STATUSES:
                self._limiter.pause(_retry_after(response, attempt))
                continue
            if response.status_code != 200:
                self._cache.put(path, response.status_code, "")
                return None
            try:
                data = response.json()
            except ValueError:
                await asyncio.sleep(2.0**attempt)
                continue
            self._cache.put(path, 200, response.text)
            return data
        raise RuntimeError(f"Giving up on {path} after {self._max_retries} attempts")

    async def aclose(self) -> None:
        await self._http.aclose()
