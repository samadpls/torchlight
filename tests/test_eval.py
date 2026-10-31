from datetime import UTC, datetime

import pytest

from torchlight.eval.batches import judge_block, judge_order, load_answers, load_grades
from torchlight.eval.generate import Answer, clean, length_groups, tool_query
from torchlight.schemas import EvalExample


def make_example(thread_id: int = 7) -> EvalExample:
    return EvalExample(
        question="Loss is NaN with AMP",
        reference_answer="Use a GradScaler.",
        needs_clarification=False,
        thread_id=thread_id,
        url=f"https://discuss.pytorch.org/t/x/{thread_id}",
        created_at=datetime(2025, 3, 1, tzinfo=UTC),
        tier="gold",
        answer_origin="verbatim",
        answer_post_numbers=[2],
        answer_authors=["ptrblck"],
        answered_by_ptrblck=True,
    )


@pytest.mark.parametrize(
    "reply",
    [
        '<|tool_call|>[{"name": "search_forum", "arguments": {"query": "amp nan"}}]<|/tool_call|>',
        'functools[{"name": "search_forum", "arguments": {"query": "amp nan"}}]',
        '[{"name": "search_forum", "arguments": "{\\"query\\": \\"amp nan\\"}"}]',
    ],
)
def test_tool_query_reads_phi_tool_calls(reply):
    assert tool_query(reply) == "amp nan"


def test_tool_query_ignores_json_inside_an_answer():
    assert tool_query('Set it in the config:\n```json\n{"name": "search_forum"}\n```') is None


def test_clean_drops_special_tokens_and_tool_calls():
    assert clean('<|tool_call|>[{"name": "x"}]<|/tool_call|>Use a scaler.<|end|>') == (
        "Use a scaler."
    )


def test_long_prompts_run_in_smaller_batches():
    groups = length_groups([300, 8186, 400, 2550, 100], max_tokens=1024)
    assert groups == [([0, 2, 4], 16), ([3], 4), ([1], 2)]


def test_api_check_flags_invented_names_in_code():
    pytest.importorskip("torch")
    from torchlight.eval.apis import invalid_names

    answer = (
        "Your `torch.rfft` call was removed in 1.8.\n"
        "```python\nscaler = torch.amp.GradScaler()\nx = F.relu(x)\noptim.step()\n"
        "opt = optim.AdamW(params)\n# old: torch.rfft(x, 1)\ntorch.nn.FlashLinear(3)\n"
        "writer = torch.utils.tensorboard.SummaryWriter()\n```"
    )

    assert invalid_names(answer) == ["torch.nn.FlashLinear"]


def test_judge_grades_map_back_to_arms(tmp_path):
    example = make_example()
    answers = {
        arm: Answer(arm=arm, thread_id=7, answer=f"answer {arm}", searches=[], generated_tokens=1)
        for arm in "ABD"
    }
    block = judge_block(example, answers)
    order = judge_order(7, list(answers))
    results = tmp_path / "results"
    results.mkdir()
    lines = [f"candidate {n}: score {2 + n} | asks no" for n in range(1, 4)]
    (results / "batch_0000.md").write_text("=== item 7 ===\n" + "\n".join(lines) + "\n")

    grades, invalid = load_grades(results, {7: sorted(answers)})

    assert invalid == 0
    assert {g.arm: g.score for g in grades} == {arm: 3 + i for i, arm in enumerate(order)}
    assert f"--- candidate 1 ---\nanswer {order[0]}" in block


def test_incomplete_judge_blocks_are_counted_invalid(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "batch_0000.md").write_text("=== item 7 ===\ncandidate 1: score 4 | asks no\n")

    grades, invalid = load_grades(results, {7: ["A", "B"]})

    assert (grades, invalid) == ([], 1)


def test_agent_answers_are_parsed(tmp_path):
    (tmp_path / "batch_0000.md").write_text(
        "=== item 7 ===\nsearches: amp nan | grad scaler\nanswer:\nUse a `GradScaler`.\n\n"
        "=== item 8 ===\nsearches:\nanswer:\nPlease share the full error.\n"
    )

    answers = load_answers(tmp_path, "C")

    assert [(a.thread_id, a.searches, a.answer) for a in answers] == [
        (7, ["amp nan", "grad scaler"], "Use a `GradScaler`."),
        (8, [], "Please share the full error."),
    ]
