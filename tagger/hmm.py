from typing import Dict, NamedTuple
from utils import invert, dict_with_missing
from toolz.curried import merge, valmap, pipe
from math import log

from resources import ud_treebank

__all__ = ["HMM", "hmm_ud_english"]


def div_by_total(counts: dict):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def div_by_total_log(counts: dict):
    to_sub = log(sum(counts.values()))
    return {k: log(v) - to_sub for k, v in counts.items()}


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

    # smoothing: dai conteggio 1 alle transizioni che non avvengono mai
    default_count = dict.fromkeys(counts.keys(), 1)
    return valmap(lambda c: merge(default_count, c), counts)


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
            # if not word.is_multiword():
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
    uses_log: bool


def train_from_conll(training_set, dev_set, use_log=True):
    _div_by_total = div_by_total_log if use_log else div_by_total

    transitions = pipe(training_set, transition_counts, valmap(_div_by_total), invert)
    emissions = pipe(training_set, emission_counts, valmap(_div_by_total), invert)
    smoothing = pipe(dev_set, smoothing_counts, _div_by_total)
    emissions = dict_with_missing(emissions, smoothing)

    return HMM(transitions, emissions, use_log)


def hmm_ud_english():
    return train_from_conll(ud_treebank("train"), ud_treebank("dev"))
