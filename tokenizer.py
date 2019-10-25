import re

TOKENIZE_RE = re.compile(r"'?\w+|[^\w\s]+")


def tokenize(sentence: str):
    return TOKENIZE_RE.findall(sentence)
