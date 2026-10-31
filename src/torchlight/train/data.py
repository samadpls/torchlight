import json
from collections.abc import Callable
from pathlib import Path

from torchlight.build.examples import SYSTEM_PROMPT
from torchlight.schemas import SFTExample

# mlx-lm reads train.jsonl and valid.jsonl from its data folder.
SPLIT_FILES = {"train": "train", "validation": "valid"}

ChatRow = list[dict[str, str]]


def chat_messages(example: SFTExample) -> ChatRow:
    system = {"role": "system", "content": SYSTEM_PROMPT}
    return [system] + [m.model_dump() for m in example.messages]


def export(
    sft_dir: Path, out_dir: Path, count_tokens: Callable[[ChatRow], int], max_tokens: int
) -> dict[str, int]:
    """Write the sft split as mlx-lm chat files.

    Rows longer than `max_tokens` are dropped rather than truncated: mlx-lm truncates from
    the end, which would cut the answer, the only part that is trained.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    for split, name in SPLIT_FILES.items():
        rows = []
        dropped = 0
        for line in (sft_dir / f"{split}.jsonl").read_text().splitlines():
            messages = chat_messages(SFTExample.model_validate_json(line))
            if count_tokens(messages) > max_tokens:
                dropped += 1
                continue
            rows.append(json.dumps({"messages": messages}, ensure_ascii=False))
        (out_dir / f"{name}.jsonl").write_text("\n".join(rows) + "\n")
        stats[f"{split}: kept"] = len(rows)
        stats[f"{split}: longer than {max_tokens} tokens"] = dropped
    return stats
