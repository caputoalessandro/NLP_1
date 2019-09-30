import pyconll
from typing import Mapping, List
from collections import defaultdict
from tagger.abc import PosTagger
from resources import ud_treebank


class BaselineTagger(PosTagger):
    def __init__(self, model):
        self.model: Mapping[str, str] = model

    def pos_tag(self, tokens: List[str]):
        return [self.model[tok] for tok in tokens]


def most_frequent_pos_for_forms(counts: Mapping[str, Mapping[str, int]]):
    return {
        form: max(pos_counts.keys(), key=lambda pos: pos_counts[pos])
        for form, pos_counts in counts.items()
    }


def ud_baseline_tagger():
    training_set = ud_treebank("train")
    counts = defaultdict(lambda: defaultdict(lambda: 0))

    for sentence in training_set:
        for token in sentence:
            counts[token.form][token.upos] += 1

    model = most_frequent_pos_for_forms(counts)
    return BaselineTagger(model)
