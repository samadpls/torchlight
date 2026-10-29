from datetime import UTC, datetime, timedelta

from torchlight.build.clean import clean_markdown, depends_on_image
from torchlight.build.examples import answer_messages, clarification_messages
from torchlight.build.resolve import resolve
from torchlight.build.review import code_is_grounded, load_results
from torchlight.schemas import Post, Thread

START = datetime(2022, 3, 1, tzinfo=UTC)


def make_thread(*posts: tuple[str, str], accepted: int | None = None) -> Thread:
    return Thread(
        id=1,
        title="Loss is NaN",
        slug="loss-is-nan",
        created_at=START,
        posts=[
            Post(
                id=100 + n,
                post_number=n,
                username=user,
                created_at=START + timedelta(hours=n),
                content=content,
                accepted_answer=n == accepted,
            )
            for n, (user, content) in enumerate(posts, start=1)
        ],
    )


CLARIFY = "Could you post a minimal, executable code snippet that reproduces the issue?"
FIX = "Your learning rate is too high. Lower it to `1e-4` and the loss stays finite."


def test_accepted_helper_post_is_the_answer():
    thread = make_thread(("asker", "My loss is NaN."), ("ptrblck", FIX), accepted=2)

    resolution = resolve(thread)

    assert (resolution.status, resolution.answer_post_number) == ("accepted", 2)


def test_accepted_thank_you_points_to_the_post_it_thanks():
    thread = make_thread(
        ("asker", "My loss is NaN."),
        ("ptrblck", FIX),
        ("asker", "Thanks, that worked!"),
        accepted=3,
    )

    resolution = resolve(thread)

    assert (resolution.status, resolution.answer_post_number) == ("accepted", 2)
    assert resolution.method == "accepted acknowledgement"


def test_asker_confirmation_without_accepted_post():
    thread = make_thread(
        ("asker", "My loss is NaN."),
        ("helper", "Try gradient clipping."),
        ("asker", "Still NaN."),
        ("ptrblck", FIX),
        ("asker", "That fixed it, thank you."),
    )

    resolution = resolve(thread)

    assert (resolution.status, resolution.answer_post_number) == ("confirmed", 4)


def test_unresolved_thread():
    thread = make_thread(("asker", "My loss is NaN."), ("ptrblck", CLARIFY))

    assert resolve(thread).status == "unresolved"


def test_multi_turn_answer_keeps_context_and_ends_with_answer():
    thread = make_thread(
        ("asker", "My loss is NaN."),
        ("ptrblck", CLARIFY),
        ("asker", "Here is my code: ```x = model(y)```"),
        ("other", "Unrelated guess."),
        ("ptrblck", FIX),
    )

    messages = answer_messages(thread, 5, FIX, single_turn=False)

    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[0].content.startswith("Loss is NaN")
    assert messages[-1].content == FIX


def test_clarification_example_uses_first_helper_question():
    thread = make_thread(
        ("asker", "My loss is NaN."), ("ptrblck", CLARIFY), ("asker", "code"), ("ptrblck", FIX)
    )

    messages = clarification_messages(thread, answer_post_number=4)

    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[-1].content == CLARIFY


def test_no_clarification_when_first_reply_is_the_answer():
    thread = make_thread(("asker", "My loss is NaN."), ("ptrblck", FIX))

    assert clarification_messages(thread, answer_post_number=2) is None


def test_clean_markdown_drops_quotes_and_images():
    raw = '[quote="a, post:1"]old text[/quote]\nSee ![shot](upload://x.png) and mail me@x.com'

    cleaned = clean_markdown(raw)

    assert cleaned == "See [image] and mail [email]"
    assert depends_on_image(cleaned)


def test_load_results_parses_blocks_and_rejects_bad_ones(tmp_path):
    (tmp_path / "batch_0000.md").write_text(
        "=== thread 11 ===\nverdict: keep\nsources: 3\nreason: fine\n\n"
        "=== thread 12 ===\nverdict: compose\nsources: 2, 4\nreason: split\nanswer:\n"
        "Lower the lr.\n\n```python\nopt = SGD(lr=1e-4)\n```\n\n"
        "=== thread 13 ===\nverdict: compose\nsources: 2\nreason: missing answer\n\n"
        "=== thread 14 ===\nverdict: maybe\n"
    )

    results, invalid = load_results(tmp_path)

    assert sorted(results) == [11, 12]
    assert results[12].source_post_numbers == [2, 4]
    assert results[12].answer.endswith("```")
    assert invalid == 2


def test_composed_code_must_come_from_sources():
    source = "Use this:\n```python\nmodel.to(device)\n```"

    assert code_is_grounded("Do it:\n```python\nmodel.to( device )\n```", [source])
    assert not code_is_grounded("```python\nmodel.cuda()\n```", [source])
