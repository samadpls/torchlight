import json
from pathlib import Path

import bm25s
from pydantic import BaseModel

DOCS_FILE = "docs.jsonl"
MAX_QUESTION_CHARS = 800
MAX_ANSWER_CHARS = 1_500


class ForumDoc(BaseModel):
    thread_id: int
    url: str
    title: str
    question: str
    answer: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.question}\n{self.answer}"


def load_docs(threads_path: Path, split: str = "train") -> list[ForumDoc]:
    """One document per resolved thread in `split`: its question and the post that resolved it."""
    docs = []
    with threads_path.open() as f:
        for line in f:
            thread = json.loads(line)
            answer_number = thread["resolution"].get("answer_post_number")
            if thread["split"] != split or thread["resolution"]["status"] == "unresolved":
                continue
            posts = {p["post_number"]: p["content"] for p in thread["posts"]}
            if answer_number not in posts or 1 not in posts:
                continue
            docs.append(
                ForumDoc(
                    thread_id=thread["thread_id"],
                    url=thread["url"],
                    title=thread["title"],
                    question=posts[1],
                    answer=posts[answer_number],
                )
            )
    return docs


def build_index(docs: list[ForumDoc], index_dir: Path) -> None:
    retriever = bm25s.BM25()
    retriever.index(
        bm25s.tokenize([d.text for d in docs], show_progress=False), show_progress=False
    )
    retriever.save(index_dir, show_progress=False)
    (index_dir / DOCS_FILE).write_text("".join(d.model_dump_json() + "\n" for d in docs))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\n[...]"


def format_hits(hits: list[ForumDoc]) -> str:
    if not hits:
        return "No matching forum threads."
    blocks = [
        f"[{n}] {d.title} ({d.url})\n"
        f"Question:\n{_clip(d.question, MAX_QUESTION_CHARS)}\n"
        f"Answer that resolved it:\n{_clip(d.answer, MAX_ANSWER_CHARS)}"
        for n, d in enumerate(hits, start=1)
    ]
    return "\n\n".join(blocks)


class ForumSearch:
    """BM25 over resolved training-split threads. This is the `search_forum` tool."""

    def __init__(self, index_dir: Path):
        self.retriever = bm25s.BM25.load(index_dir, show_progress=False, verbosity=0)
        self.docs = [
            ForumDoc.model_validate_json(line)
            for line in (index_dir / DOCS_FILE).read_text().splitlines()
        ]

    def search(self, query: str, k: int = 3) -> list[ForumDoc]:
        tokens = bm25s.tokenize([query], show_progress=False)
        if not tokens.vocab:
            return []
        ids, scores = self.retriever.retrieve(tokens, k=k, show_progress=False)
        return [self.docs[i] for i, score in zip(ids[0], scores[0], strict=True) if score > 0]
