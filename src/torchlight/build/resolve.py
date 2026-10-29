import re

from torchlight.build.clean import has_code
from torchlight.schemas import Post, Resolution, Thread

POSITIVE = re.compile(
    r"\b(it|that|this|which|now it|everything)\s+(works?|worked|fixed|solved|helped|did the trick)"
    r"|\bworks?\s+(now|fine|perfectly|great|well)\b"
    r"|\b(problem|issue|error)\s+(is\s+)?(solved|fixed|resolved|gone)\b"
    r"|\b(solved|fixed|resolved)\s+(it|the|my)\b"
    r"|\bthank(s| you)[^.!\n]{0,40}\b(works?|worked|solved|fixed|helped)\b",
    re.IGNORECASE,
)
NEGATIVE = re.compile(
    r"\b(still|doesn'?t|does not|didn'?t|did not|isn'?t|not work|no luck|same error|but now)\b",
    re.IGNORECASE,
)
ACKNOWLEDGEMENT = re.compile(r"\b(thanks?|thank you|thx|solved|works?|worked)\b", re.IGNORECASE)
REFERS_ELSEWHERE = re.compile(
    r"\b(as (i|you) (said|mentioned|suggested)|mentioned above|see above|above post"
    r"|previous (post|reply|answer)|my (last|earlier|previous) (post|reply)|in addition to"
    r"|combined with|as suggested by)\b",
    re.IGNORECASE,
)
CLARIFYING = re.compile(
    r"\b(could|can|would) you (please )?(post|share|provide|add|check|print|explain|describe|"
    r"give|try|show|update)"
    r"|minimal[^.?\n]{0,30}(code|example|snippet)|executable|reproduc"
    r"|full (error|stack ?trace|traceback)|which (pytorch|cuda|torch) version"
    r"|what (is|are|does) the (shape|output|error|exact)",
    re.IGNORECASE,
)
MENTION = re.compile(r"(?<![\w.])@[A-Za-z0-9_][\w.-]*")
LINK = re.compile(r"https?://")
SHORT_ANSWER_CHARS = 60
QUESTION_ANSWER_CHARS = 300
LINK_ANSWER_CHARS = 200


def is_confirmation(post: Post) -> bool:
    return bool(POSITIVE.search(post.content)) and not NEGATIVE.search(post.content)


def is_acknowledgement(post: Post) -> bool:
    """A short thank-you post that points at the real answer rather than being it."""
    return (
        len(post.content) < 400
        and not has_code(post.content)
        and bool(ACKNOWLEDGEMENT.search(post.content))
    )


def is_clarifying_question(post: Post) -> bool:
    return "?" in post.content and bool(CLARIFYING.search(post.content))


def _answered_post(thread: Thread, ack: Post) -> Post | None:
    """The helper post an acknowledgement refers to: its reply target, else the latest before."""
    if ack.reply_to_post_number:
        target = thread.post(ack.reply_to_post_number)
        if target is not None and target.username != thread.asker:
            return target
    earlier = [
        p
        for p in thread.posts
        if p.post_number < ack.post_number and p.post_number > 1 and p.username != thread.asker
    ]
    return earlier[-1] if earlier else None


def _review_reasons(thread: Thread, answer: Post) -> list[str]:
    text = answer.content.strip()
    reasons = []
    if REFERS_ELSEWHERE.search(text):
        reasons.append("refers to other posts")
    if not has_code(text):
        if len(text) < QUESTION_ANSWER_CHARS and text.endswith("?"):
            reasons.append("answer is a question")
        elif len(text) < SHORT_ANSWER_CHARS:
            reasons.append("very short answer")
        elif len(text) < LINK_ANSWER_CHARS and LINK.search(text):
            reasons.append("mostly a link")
    if answer.username == thread.asker and MENTION.search(text):
        reasons.append("asker's fix credits another post")
    return reasons


def resolve(thread: Thread) -> Resolution:
    accepted = next((p for p in thread.posts if p.accepted_answer and p.post_number > 1), None)
    if accepted is not None:
        if accepted.username == thread.asker and is_acknowledgement(accepted):
            target = _answered_post(thread, accepted)
            if target is None:
                return Resolution(status="unresolved", method="accepted acknowledgement only")
            return Resolution(
                status="accepted",
                answer_post_number=target.post_number,
                method="accepted acknowledgement",
                review_reasons=_review_reasons(thread, target),
            )
        return Resolution(
            status="accepted",
            answer_post_number=accepted.post_number,
            method="accepted",
            review_reasons=_review_reasons(thread, accepted),
        )

    for post in thread.posts[1:]:
        if post.username == thread.asker and is_confirmation(post):
            target = _answered_post(thread, post)
            if target is not None:
                return Resolution(
                    status="confirmed",
                    answer_post_number=target.post_number,
                    method="asker confirmation",
                    review_reasons=_review_reasons(thread, target),
                )
    return Resolution(status="unresolved", method="no resolution signal")
