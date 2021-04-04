from functools import cached_property
import pyconll


class Corpus:
    def __init__(self, name):
        self.name = name

    def _load(self, kind):
        return pyconll.load_from_file(f"resources/{self.name}-ud-{kind}.conllu")

    @cached_property
    def train(self):
        return self._load("train")

    @cached_property
    def test(self):
        return self._load("test")

    @cached_property
    def dev(self):
        return self._load("dev")

    @classmethod
    def latin(cls):
        return cls('la_llct')

    @classmethod
    def greek(cls):
        return cls('grc_perseus')


POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
            "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
