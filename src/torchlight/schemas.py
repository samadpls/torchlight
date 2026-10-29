from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from torchlight.scrape import FORUM_URL


class Post(BaseModel):
    id: int
    post_number: int
    username: str
    created_at: datetime
    content: str
    reply_to_post_number: int | None = None
    accepted_answer: bool = False
    like_count: int = 0


class Thread(BaseModel):
    id: int
    title: str
    slug: str
    created_at: datetime
    category_id: int | None = None
    tags: list[str] = Field(default_factory=list)
    posts: list[Post]

    @property
    def url(self) -> str:
        return f"{FORUM_URL}/t/{self.slug}/{self.id}"

    @property
    def asker(self) -> str:
        return self.posts[0].username

    def post(self, post_number: int) -> Post | None:
        return next((p for p in self.posts if p.post_number == post_number), None)


ResolutionStatus = Literal["accepted", "confirmed", "unresolved"]


class Resolution(BaseModel):
    status: ResolutionStatus
    answer_post_number: int | None = None
    method: str
    review_reasons: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """What a reviewing agent returns for one flagged thread."""

    thread_id: int
    verdict: Literal["keep", "compose", "drop"]
    source_post_numbers: list[int] = Field(default_factory=list)
    answer: str | None = None
    reason: str = ""


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SFTExample(BaseModel):
    messages: list[Message]
    kind: Literal["answer", "clarification"]
    thread_id: int
    url: str
    created_at: datetime
    tier: Literal["gold", "silver"]
    answer_origin: Literal["verbatim", "composed"]
    answer_post_numbers: list[int]
    answer_authors: list[str]


class EvalExample(BaseModel):
    question: str
    reference_answer: str
    needs_clarification: bool
    thread_id: int
    url: str
    created_at: datetime
    tier: Literal["gold", "silver"]
    answer_origin: Literal["verbatim", "composed"]
    answer_post_numbers: list[int]
    answer_authors: list[str]
    answered_by_ptrblck: bool
