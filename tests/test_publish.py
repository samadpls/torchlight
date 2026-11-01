import re

import pytest

from torchlight.publish.card import kaggle_run_text, mlx_run_text, model_card, repo_names
from torchlight.train import LoRARun
from torchlight.train.log import loss_table, read_log

LOG = (
    "Trainable parameters: 0.301% (11.534M/3836.022M)\n"
    "  iter   train_loss     tok/s     tokens\n"
    "     1    val 3.280    266.18s\n"
    "    10    4.719 ▼       25      2.0k\r    20    4.405 ▼       34      4.0k\n"
    "\x1b[1m    30    2.000 ▼       42      6.1k\x1b[0m\n"
    "    30    val 1.900    120.00s\n"
    "    40    1.000 ▼       41      7.8k\n"
)


def test_log_reads_losses_and_trainable_parameters(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(LOG)

    log = read_log(path)

    assert [(p.batch, p.loss) for p in log.train] == [
        (10, 4.719),
        (20, 4.405),
        (30, 2.0),
        (40, 1.0),
    ]
    assert [(p.batch, p.loss) for p in log.validation] == [(0, 3.28), (30, 1.9)]
    assert (log.trainable_millions, log.trainable_percent) == (11.534, 0.301)


def test_loss_table_averages_training_loss_between_rows(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(LOG)

    table = loss_table(read_log(path), batches_per_epoch=40, every=20).splitlines()

    assert table[2:] == [
        "| 0.00 | 0 | - | 3.280 |",
        "| 0.50 | 20 | 4.562 | - |",
        "| 0.75 | 30 | 2.000 | 1.900 |",
        "| 1.00 | 40 | 1.000 | - |",
    ]


FACTS = dict.fromkeys(
    [
        "train_rows",
        "dropped",
        "validation_rows",
        "eval_count",
        "clarify_count",
        "answerable_count",
        "search_docs",
        "batches",
        "steps",
        "warmup_updates",
    ],
    1,
) | {
    "valid_since": "2024-07-01",
    "eval_since": "2025-01-01",
    "eval_count_since": "2025-04-01",
    "eval_until": "2026-10-08",
    "hours": 6.0,
    "trainable": "11.5M",
    "trainable_share": "0.30%",
    "size": "3.8B",
    "optimizer": "AdamW",
    "mlx_lm": "0.32.2",
    "mlx": "0.32.4",
    "gpu": "Tesla T4",
    "unsloth": "2026.10.11",
    "transformers": "5.16.1",
}
PLACEHOLDER = re.compile(r"\{[a-z_]+(:[,.0-9a-z]*)?\}")


@pytest.mark.parametrize("kind", ["model", "lora", "gguf"])
def test_mlx_model_cards_fill_every_field(kind):
    params = LoRARun()
    card = model_card(
        kind,
        "samadpls/torchlight-phi4-mini",
        params,
        FACTS | mlx_run_text(params, FACTS),
        "| loss |",
        "| results |",
        gguf_files={"torchlight-phi4-mini-Q4_K_M.gguf": 2_500_000_000},
    )

    assert card.startswith("---\nlicense: mit\n")
    assert "resolve/main/assets/hero.png" in card
    assert repo_names("samadpls/torchlight-phi4-mini")[kind] in card
    assert not PLACEHOLDER.search(card)
    if kind == "gguf":
        assert "| `torchlight-phi4-mini-Q4_K_M.gguf` | 2.5 GB | Smaller" in card
    if kind == "lora":
        assert "--adapter-path torchlight-lora/mlx" in card


@pytest.mark.parametrize("kind", ["model", "lora", "gguf"])
def test_kaggle_model_cards_use_the_base_license_and_skip_mlx(kind):
    params = LoRARun(model="Qwen/Qwen3-8B", bits=4, layers=-1)
    card = model_card(
        kind,
        "samadpls/torchlight-qwen3-8b",
        params,
        FACTS | kaggle_run_text(params, FACTS, ["English only."]),
        "| loss |",
        "| results |",
        gguf_files={"torchlight-qwen3-8b-Q4_K_M.gguf": 5_030_000_000},
    )

    assert card.startswith("---\nlicense: apache-2.0\n")
    assert "Apache-2.0" in card and "mlx/" not in card
    assert not PLACEHOLDER.search(card)
