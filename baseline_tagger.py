import pyconll
from typing import Mapping, List
from collections import defaultdict, Counter
from pos_tagger import PosTagger


class BaselineTagger(PosTagger):
    def __init__(self, model):
        self.model: Mapping[str, str] = model

    def pos_tag(self, tokens: List[str]):
        return [self.model[tok] for tok in tokens]


def most_frequent_pos_for_forms(counts: Mapping[str, Counter]):
    return {
        form: pos_counts.most_common(1)[0][0]
        for form, pos_counts in counts.items()
    }


def ud_baseline_tagger():
    training_set = pyconll.load_from_file(
        "resources/en_partut-ud-train.conllu"
    )

    counts = defaultdict(Counter)

    for sentence in training_set:
        for token in sentence:
            counts[token.form][token.upos] += 1

    model = most_frequent_pos_for_forms(counts)
    return BaselineTagger(model)
