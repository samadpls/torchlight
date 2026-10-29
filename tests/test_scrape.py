import asyncio

import httpx

from torchlight.scrape import ForumClient, ResponseCache, remaining_posts_paths


def test_remaining_posts_paths_chunks_missing_ids():
    topic = {
        "id": 7,
        "post_stream": {
            "posts": [{"id": i} for i in range(1, 21)],
            "stream": list(range(1, 46)),
        },
    }

    paths = remaining_posts_paths(topic)

    assert len(paths) == 2
    assert paths[0].startswith("/t/7/posts.json?post_ids[]=21&")
    assert paths[0].count("post_ids[]") == 20
    assert paths[1].count("post_ids[]") == 5
    assert all(p.endswith("&include_raw=1") for p in paths)


def test_client_retries_429_then_caches(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"ok": True})

    async def run():
        cache = ResponseCache(tmp_path / "cache.sqlite")
        client = ForumClient(cache, rate=1000, base_url="https://forum.test")
        client._http = httpx.AsyncClient(
            base_url="https://forum.test", transport=httpx.MockTransport(handler)
        )
        first = await client.get_json("/t/1.json")
        second = await client.get_json("/t/1.json")
        await client.aclose()
        return first, second

    first, second = asyncio.run(run())

    assert first == second == {"ok": True}
    assert len(calls) == 2


def test_client_caches_404_as_none(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async def run():
        cache = ResponseCache(tmp_path / "cache.sqlite")
        client = ForumClient(cache, rate=1000, base_url="https://forum.test")
        client._http = httpx.AsyncClient(
            base_url="https://forum.test", transport=httpx.MockTransport(handler)
        )
        result = await client.get_json("/t/404.json")
        await client.aclose()
        return result, cache.get("/t/404.json")

    result, cached = asyncio.run(run())

    assert result is None
    assert cached == (404, "")
