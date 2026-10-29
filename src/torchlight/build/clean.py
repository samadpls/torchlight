import re

QUOTE = re.compile(r"\[quote[^\]]*\].*?\[/quote\]\s*", re.DOTALL | re.IGNORECASE)
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_IMAGE = re.compile(r"<img[^>]*>", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
SECRET = re.compile(
    r"\b(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})\b"
)
BLANK_LINES = re.compile(r"\n{3,}")
CODE_FENCE = re.compile(r"```")
IMAGE_PLACEHOLDER = "[image]"

OUTDATED_API = re.compile(
    r"volatile\s*=\s*True|\.data\[0\]|async\s*=\s*True|torch\.autograd\.Variable|\bVariable\("
)


def clean_markdown(raw: str) -> str:
    text = QUOTE.sub("", raw)
    text = MARKDOWN_IMAGE.sub(IMAGE_PLACEHOLDER, text)
    text = HTML_IMAGE.sub(IMAGE_PLACEHOLDER, text)
    text = SECRET.sub("[redacted]", text)
    text = EMAIL.sub("[email]", text)
    return BLANK_LINES.sub("\n\n", text).strip()


def has_code(text: str) -> bool:
    return bool(CODE_FENCE.search(text)) or any(
        line.startswith("    ") and line.strip() for line in text.splitlines()
    )


def depends_on_image(text: str) -> bool:
    """True when an image carries the question: little prose and no code besides the image."""
    if IMAGE_PLACEHOLDER not in text or has_code(text):
        return False
    prose = text.replace(IMAGE_PLACEHOLDER, "")
    return len(prose.split()) < 60


def is_mostly_non_latin(text: str, threshold: float = 0.3) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(not c.isascii() for c in letters) / len(letters) > threshold


def uses_outdated_api(text: str) -> bool:
    return bool(OUTDATED_API.search(text))
