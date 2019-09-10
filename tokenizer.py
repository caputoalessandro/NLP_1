import re

TOKENIZE_RE = re.compile(r"\w+|[^\w\s]+")


def tokenize(sentence: str):
    return [tok for tok in TOKENIZE_RE.split(sentence) if tok]
