from collections import defaultdict
from typing import Mapping, List

from resources import Corpus
from tagger.abc import PosTagger


def create_baseline_model(counts: Mapping[str, Mapping[str, int]]):
    model = {
        form: max(pos_counts.keys(), key=lambda pos: pos_counts[pos])
        for form, pos_counts in counts.items()
    }
    return model


class BaselineTagger(PosTagger):
    def __init__(self, model):
        self.model: Mapping[str, str] = model
        self.missing = None

    def pos_tags(self, tokens: List[str]):
        return [self.model.get(tok, self.missing) for tok in tokens]

    def with_default_for_missing(self, pos):
        b = BaselineTagger(self.model)
        b.missing = pos
        return b

    @classmethod
    def train(cls, corpus: Corpus):
        counts = defaultdict(lambda: defaultdict(lambda: 0))

        for sentence in corpus.train:
            for token in sentence:
                counts[token.form][token.upos] += 1

        model = create_baseline_model(counts)
        return cls(model)
