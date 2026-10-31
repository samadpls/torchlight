import importlib
import re
from functools import cache

# Only fenced blocks, without comments: inline code and comments often quote the asker's
# old code to explain why it no longer works.
CODE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
COMMENT = re.compile(r"#.*")
# nn, F and optim are the aliases nearly every forum answer imports torch modules under.
NAME = re.compile(r"(?<![\w.])(torch|nn|F|optim)((?:\.[A-Za-z_]\w*)+)")
ALIASES = {"torch": "torch", "nn": "torch.nn", "F": "torch.nn.functional", "optim": "torch.optim"}
OPTIM_SUBMODULES = {"lr_scheduler", "swa_utils"}


def _is_optimizer_variable(alias: str, rest: str) -> bool:
    # `optim` is also a common name for an optimizer instance, as in `optim.step()`.
    attribute = rest.split(".")[1]
    return alias == "optim" and not (attribute[0].isupper() or attribute in OPTIM_SUBMODULES)


def torch_names(answer: str) -> set[str]:
    """Fully qualified torch names used in the answer's code blocks."""
    names = set()
    for block in CODE.findall(answer):
        for alias, rest in NAME.findall(COMMENT.sub("", block)):
            if not _is_optimizer_variable(alias, rest):
                names.add(ALIASES[alias] + rest)
    return names


@cache
def resolves(name: str) -> bool:
    parts = name.split(".")
    obj = importlib.import_module(parts[0])
    for i, part in enumerate(parts[1:], start=1):
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        try:
            obj = importlib.import_module(".".join(parts[: i + 1]))
        except ModuleNotFoundError as error:
            # torch.utils.tensorboard and friends exist but need an optional package.
            return bool(error.name) and not error.name.startswith("torch")
        except ImportError:
            return False
    return True


def invalid_names(answer: str) -> list[str]:
    return sorted(name for name in torch_names(answer) if not resolves(name))
