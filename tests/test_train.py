import json
from datetime import UTC, datetime

from torchlight.build.examples import SYSTEM_PROMPT
from torchlight.schemas import Message, SFTExample
from torchlight.train import LoRARun, export


def make_example(answer: str) -> SFTExample:
    return SFTExample(
        messages=[
            Message(role="user", content="Loss is NaN"),
            Message(role="assistant", content=answer),
        ],
        kind="answer",
        thread_id=1,
        url="https://discuss.pytorch.org/t/loss-is-nan/1",
        created_at=datetime(2022, 3, 1, tzinfo=UTC),
        tier="gold",
        answer_origin="verbatim",
        answer_post_numbers=[2],
        answer_authors=["ptrblck"],
    )


def write_split(path, *examples: SFTExample) -> None:
    path.write_text("".join(e.model_dump_json() + "\n" for e in examples))


def test_export_prepends_system_prompt_and_drops_long_rows(tmp_path):
    sft = tmp_path / "sft"
    sft.mkdir()
    write_split(sft / "train.jsonl", make_example("short"), make_example("x" * 500))
    write_split(sft / "validation.jsonl", make_example("short"))

    def count_chars(messages):
        return sum(len(m["content"]) for m in messages)

    stats = export(sft, tmp_path / "out", count_chars, max_tokens=400)

    lines = (tmp_path / "out" / "train.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert stats == {
        "train: kept": 1,
        "train: longer than 400 tokens": 1,
        "validation: kept": 1,
        "validation: longer than 400 tokens": 0,
    }
    assert list(rows[0]) == ["messages"]
    assert rows[0]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "short"}
    assert (tmp_path / "out" / "valid.jsonl").exists()


def test_schedule_counts_optimizer_updates(tmp_path):
    run = LoRARun(epochs=2, batch_size=2, grad_accumulation=4, warmup=0.05)

    config = run.mlx_config("base", tmp_path / "data", tmp_path / "adapters", train_rows=1000)

    assert config["iters"] == 1000
    assert config["lr_schedule"] == {
        "name": "cosine_decay",
        "warmup": 12,
        "arguments": [2e-4, 238, 0.0],
    }
    assert config["lora_parameters"]["scale"] == 2.0


def test_trial_iters_override_epochs(tmp_path):
    config = LoRARun().mlx_config("base", tmp_path, tmp_path, train_rows=1000, iters=100)

    assert config["iters"] == 100
    assert config["lr_schedule"]["arguments"][1] == 25 - 1
