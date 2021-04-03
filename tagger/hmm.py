from collections import Counter, defaultdict
from math import log
from typing import Dict, NamedTuple

from toolz.curried import merge, valmap, pipe

from resources import Corpus
from utils import transpose, dict_with_missing

__all__ = ["HMM"]


def div_by_total_log(counts: dict):
    denom = log(sum(counts.values()))
    return {k: log(v) - denom for k, v in counts.items()}


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


def emissions_smoothing_always_noun(_):
    ret = defaultdict(lambda: log(.0))
    ret['NOUN'] = log(1)
    return ret


def emissions_smoothing_noun_or_verb(_):
    ret = defaultdict(lambda: log(.0))
    ret['NOUN'] = log(.5)
    ret['VERB'] = log(.5)
    return ret


def emissions_smoothing_occurring_once(dev_set):
    word_to_pos = {}

    for sentence in dev_set:
        for word in sentence:
            if word.form in word_to_pos:
                word_to_pos[word.form] = None
            else:
                word_to_pos[word.form] = word.upos

    return div_by_total_log(Counter(pos for pos in word_to_pos.values() if pos is not None))


class HMM(NamedTuple):
    transitions: Dict[str, Dict[str, float]]
    emissions: Dict[str, Dict[str, float]]

    @classmethod
    def train(cls, corpus: Corpus, emissions_smoothing=emissions_smoothing_occurring_once):
        transitions = pipe(
            corpus.train,
            transition_counts,
            transitions_smoothing,
            valmap(div_by_total_log),
            transpose,
        )
        emissions = pipe(corpus.train, emission_counts, valmap(div_by_total_log), transpose)
        em_smoothing = emissions_smoothing(corpus.dev)

        emissions = dict_with_missing(emissions, em_smoothing)

        return cls(transitions, emissions)



