from collections import defaultdict
from typing import Mapping, List

from resources import ud_treebank
from tagger.abc import PosTagger
from utils import DictWithMissing


class BaselineTagger(PosTagger):
    def __init__(self, model):
        self.model: Mapping[str, str] = model

    def pos_tags(self, tokens: List[str]):
        return [self.model[tok] for tok in tokens]


def create_baseline_model(counts: Mapping[str, Mapping[str, int]]):
    model = DictWithMissing(
        (form, max(pos_counts.keys(), key=lambda pos: pos_counts[pos]))
        for form, pos_counts in counts.items()
    )
    model.missing = 'PROPN'
    return model


def ud_baseline_tagger():
    training_set = ud_treebank("train")
    counts = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in training_set:
        for token in sentence:
            counts[token.form][token.upos] += 1

    model = create_baseline_model(counts)
    return BaselineTagger(model)

