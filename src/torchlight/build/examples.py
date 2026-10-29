from torchlight.build.resolve import is_acknowledgement, is_clarifying_question
from torchlight.schemas import Message, Post, Thread

SYSTEM_PROMPT = (
    "You are a careful PyTorch helper. Give specific, correct guidance with code where it helps. "
    "If a question lacks the details needed to answer it, ask for them, for example a minimal "
    "executable code snippet or the full error message, instead of guessing."
)
MAX_MESSAGES = 8
MAX_CHARS = 12_000
MAX_CLARIFICATION_CHARS = 1_500


def question_text(thread: Thread) -> str:
    return f"{thread.title}\n\n{thread.posts[0].content}"


def _role(thread: Thread, post: Post) -> str:
    return "user" if post.username == thread.asker else "assistant"


def _turns(thread: Thread, posts: list[Post]) -> list[Message]:
    messages: list[Message] = []
    for post in posts:
        role = _role(thread, post)
        content = question_text(thread) if post.post_number == 1 else post.content
        if messages and messages[-1].role == role:
            messages[-1] = Message(role=role, content=f"{messages[-1].content}\n\n{content}")
        else:
            messages.append(Message(role=role, content=content))
    return messages


def _within_limits(messages: list[Message]) -> bool:
    return len(messages) <= MAX_MESSAGES and sum(len(m.content) for m in messages) <= MAX_CHARS


def answer_messages(
    thread: Thread, answer_post_number: int, answer: str, single_turn: bool
) -> list[Message] | None:
    """System prompt, the conversation up to the answer, then the answer as the final turn.

    Helper posts the asker never replied to right before the answer are left out, so the
    final assistant turn holds only the answer.
    """
    if single_turn:
        context = [thread.posts[0]]
    else:
        context = [
            p
            for p in thread.posts
            if p.post_number < answer_post_number
            and not (p.username == thread.asker and p.post_number > 1 and is_acknowledgement(p))
        ]
        while context and _role(thread, context[-1]) == "assistant":
            context.pop()
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        *_turns(thread, context),
        Message(role="assistant", content=answer),
    ]
    return messages if _within_limits(messages) else None


def first_helper_reply(thread: Thread) -> Post | None:
    return next((p for p in thread.posts[1:] if p.username != thread.asker), None)


def clarification_messages(thread: Thread, answer_post_number: int) -> list[Message] | None:
    """The asker's opening post(s) and a first reply that asks for missing details."""
    reply = first_helper_reply(thread)
    if (
        reply is None
        or reply.post_number == answer_post_number
        or len(reply.content) > MAX_CLARIFICATION_CHARS
        or not is_clarifying_question(reply)
    ):
        return None
    opening = [p for p in thread.posts if p.post_number < reply.post_number]
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        *_turns(thread, opening),
        Message(role="assistant", content=reply.content),
    ]


def needs_clarification(thread: Thread, answer_post_number: int) -> bool:
    return clarification_messages(thread, answer_post_number) is not None
