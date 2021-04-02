from math import log
from typing import Dict, NamedTuple

from toolz.curried import merge, valmap, pipe

from resources import Corpus
from utils import transpose, dict_with_missing

__all__ = ["HMM", "train_hmm"]


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


def smooth_transitions(counts):
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


def smoothing_counts(dev_set):
    smoothing_dict = {}
    count_dict = {}

    # conto occorrenze parole
    for sentence in dev_set:
        for word in sentence:
            count_dict.setdefault(word.form, 0)
            count_dict[word.form] += 1

    # conto quante volte occorre un pos solo per le parole cche appaiono una volta
    for sentence in dev_set:
        for word in sentence:
            if count_dict[word.form] == 1 and not word.is_multiword():
                smoothing_dict.setdefault(word.upos, 0)
                smoothing_dict[word.upos] += 1

    return smoothing_dict


class HMM(NamedTuple):
    transitions: Dict[str, Dict[str, float]]
    emissions: Dict[str, Dict[str, float]]


def train_hmm(corpus):
    transitions = pipe(
        corpus.train,
        transition_counts,
        smooth_transitions,
        valmap(div_by_total_log),
        transpose,
    )
    emissions = pipe(corpus.train, emission_counts, valmap(div_by_total_log), transpose)
    smoothing = pipe(corpus.dev, smoothing_counts, div_by_total_log)
    emissions = dict_with_missing(emissions, smoothing)

    return HMM(transitions, emissions)



