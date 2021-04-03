from typing import NamedTuple

from toolz.curried import merge, valmap, pipe

from resources import Corpus
from utils import transpose, DictWithMissing, counts_to_log_likelihood

__all__ = ["HMM"]


def transition_counts(training_set):
    """
    Restituisce un dizionario che contiene i conteggi delle transizioni da
    un elemento a un altro.
    """
    counts = {"Q0": {}}

    for sentence in training_set:
        sentence = [word for word in sentence if not word.is_multiword()]

        counts["Q0"].setdefault(sentence[0].upos, 0)
        counts["Q0"][sentence[0].upos] += 1
        for t1, t2 in zip(sentence, sentence[1:]):
            counts.setdefault(t1.upos, {})
            counts[t1.upos].setdefault(t2.upos, 0)
            counts[t1.upos][t2.upos] += 1
        counts[sentence[-1].upos].setdefault("Qf", 0)
        counts[sentence[-1].upos]["Qf"] += 1

    return counts


def transitions_smoothing(counts):
    # smoothing: dai conteggio 1 alle transizioni che non avvengono mai
    default_count = dict.fromkeys(counts.keys(), 1)
    return {k: merge(default_count, v) for k, v in counts.items()}


def emission_counts(training_set):
    counts = {}

    for sentence in training_set:
        for word in sentence:
            if word.upos is None:
                continue
            counts.setdefault(word.upos, {})
            counts[word.upos].setdefault(word.form, 0)
            counts[word.upos][word.form] += 1

    return counts


class HMM(NamedTuple):
    transitions: dict[str, dict[str, float]]
    emissions: DictWithMissing[str, dict[str, float]]

    def with_unknown_emissions(self, ue):
        return HMM(self.transitions, self.emissions.with_missing(ue))

    @classmethod
    def train(cls, corpus: Corpus):
        transitions = pipe(
            corpus.train,
            transition_counts,
            transitions_smoothing,
            valmap(counts_to_log_likelihood),
            transpose,
        )
        emissions = pipe(corpus.train, emission_counts, valmap(counts_to_log_likelihood), transpose, DictWithMissing)

        return cls(transitions, emissions)



