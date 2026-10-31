import ast
import json
import re
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from torchlight.build.examples import SYSTEM_PROMPT
from torchlight.retrieval import ForumSearch, format_hits
from torchlight.schemas import EvalExample

TOOL_PROMPT = (
    " You can call search_forum to look up solved threads from the PyTorch forums when it "
    "would help, then answer the user directly."
)
SEARCH_TOOL = {
    "name": "search_forum",
    "description": "Search solved PyTorch forum threads. Returns the closest questions with "
    "the answers that resolved them.",
    "parameters": {
        "query": {"description": "Keywords or an error message to search for.", "type": "str"}
    },
}
# Prompt plus new tokens held in the KV cache at once: about 3 GB for Phi-4-mini.
TOKEN_BUDGET = 24_576
MAX_BATCH = 16
# Phi-4-mini wraps calls in <|tool_call|> tokens; vLLM's parser for it also accepts functools[...].
TOOL_CALL = re.compile(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)|functools(\[.*\])", re.DOTALL)
# Without those tokens, the base model often writes the call as JSON after a sentence such as
# "let me search", or as a Python call in a code block.
JSON_CALL = re.compile(r'(\[\s*)?\{\s*"name"\s*:\s*"search_forum"')
PYTHON_CALL = re.compile(
    r"""search_forum\(\s*(?:query\s*=\s*)?("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')"""
)
SPECIAL_TOKEN = re.compile(r"<\|[a-z_/]+\|>")


class Answer(BaseModel):
    arm: str
    thread_id: int
    answer: str
    searches: list[str]
    generated_tokens: int


def _calls(payload: Any) -> Iterator[tuple[str | None, Any]]:
    for item in payload if isinstance(payload, list) else [payload]:
        if not isinstance(item, dict):
            continue
        function = item.get("function", item)
        arguments = function.get("arguments", function.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                continue
        yield function.get("name"), arguments


def _json_query(candidate: str) -> str | None:
    starts = [i for i in (candidate.find("["), candidate.find("{")) if i >= 0]
    if not starts:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(candidate[min(starts) :])
    except ValueError:
        return None
    for name, arguments in _calls(payload):
        if name == "search_forum" and isinstance(arguments, dict) and arguments.get("query"):
            return str(arguments["query"])
    return None


def tool_query(text: str) -> str | None:
    """The query of a search_forum call in the model's reply, if it made one."""
    candidates = [m.group(1) or m.group(2) for m in TOOL_CALL.finditer(text)]
    candidates += [text[m.start() :] for m in JSON_CALL.finditer(text)]
    for candidate in candidates:
        if query := _json_query(candidate):
            return query
    if match := PYTHON_CALL.search(text):
        return str(ast.literal_eval(match[1]))
    return None


def clean(text: str) -> str:
    """The answer without special tokens or tool calls."""
    text = TOOL_CALL.sub("", text)
    while match := JSON_CALL.search(text):
        try:
            _, end = json.JSONDecoder().raw_decode(text, match.start())
        except ValueError:
            break
        text = text[: match.start()] + text[end:]
    return SPECIAL_TOKEN.sub("", text).strip()


def system_message(with_tool: bool) -> dict[str, str]:
    if not with_tool:
        return {"role": "system", "content": SYSTEM_PROMPT}
    return {
        "role": "system",
        "content": SYSTEM_PROMPT + TOOL_PROMPT,
        "tools": json.dumps([SEARCH_TOOL]),
    }


def length_groups(lengths: list[int], max_tokens: int) -> list[tuple[list[int], int]]:
    """Indices grouped by prompt length, each with the number of sequences to keep in flight.
    batch_generate pads every sequence to the longest one in flight, so one 8k-token question
    among short ones multiplies the memory and time of the whole batch."""
    groups: dict[int, list[int]] = {}
    for i, length in enumerate(lengths):
        groups.setdefault(max(512, 1 << (length - 1).bit_length()), []).append(i)
    return [
        (indices, max(1, min(MAX_BATCH, TOKEN_BUDGET // (ceiling + max_tokens))))
        for ceiling, indices in sorted(groups.items())
    ]


def _generate(
    model: Any, tokenizer: Any, prompts: list[list[int]], max_tokens: int
) -> list[tuple[str, list[int]]]:
    from mlx_lm.generate import batch_generate

    results: list[tuple[str, list[int]]] = [("", [])] * len(prompts)
    for indices, size in length_groups([len(p) for p in prompts], max_tokens):
        response = batch_generate(
            model,
            tokenizer,
            [prompts[i] for i in indices],
            max_tokens=max_tokens,
            return_token_ids=True,
            completion_batch_size=size,
            prefill_batch_size=min(size, 8),
        )
        for i, text, token_ids in zip(indices, response.texts, response.token_ids, strict=True):
            results[i] = (text, token_ids)
    return results


def answer_all(
    model: Any,
    tokenizer: Any,
    arm: str,
    examples: list[EvalExample],
    search: ForumSearch | None,
    max_tokens: int,
    max_searches: int = 2,
) -> list[Answer]:
    """Answer every question greedily, in batches. With `search`, a reply that calls
    search_forum gets the results back and the model answers again, at most `max_searches`
    times; a call after that is recorded as the answer, which then reads as empty."""
    chats = [
        [system_message(search is not None), {"role": "user", "content": e.question}]
        for e in examples
    ]
    searches: list[list[str]] = [[] for _ in examples]
    generated = [0] * len(examples)
    replies = [""] * len(examples)
    pending = list(range(len(examples)))
    for round_number in range(max_searches + 1):
        prompts = [
            tokenizer.apply_chat_template(chats[i], add_generation_prompt=True, return_dict=False)
            for i in pending
        ]
        still_searching = []
        for i, (text, token_ids) in zip(
            pending, _generate(model, tokenizer, prompts, max_tokens), strict=True
        ):
            generated[i] += len(token_ids)
            replies[i] = text
            query = tool_query(text) if search else None
            if query is None or round_number == max_searches:
                continue
            searches[i].append(query)
            chats[i] += [
                {"role": "assistant", "content": text},
                {"role": "tool_response", "content": format_hits(search.search(query))},
            ]
            still_searching.append(i)
        pending = still_searching
        if not pending:
            break
    return [
        Answer(
            arm=arm,
            thread_id=e.thread_id,
            answer=clean(replies[i]),
            searches=searches[i],
            generated_tokens=generated[i],
        )
        for i, e in enumerate(examples)
    ]
