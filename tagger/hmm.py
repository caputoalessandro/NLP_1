from typing import Dict, NamedTuple
from toolz import merge, valmap

from resources import ud_treebank
from tagger.smoothing import smoothing

__all__ = ["HMM", "hmm_ud_english"]


def normalize(counts: dict):
    result = {}
    for outer_key, inner_dict in counts.items():
        denom = sum(inner_dict.values())
        result[outer_key] = {key: value / denom for key, value in inner_dict.items()}
    return result


def smooth_transitions(counts):
    default_value = dict.fromkeys(counts.keys(), 1)
    return valmap(lambda d: merge(default_value, d), counts)


def get_transition_frequencies(training_set):
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

    return normalize(smooth_transitions(counts))


def get_emission_frequencies(training_set):
    counts = {}

    for sentence in training_set:
        for word in sentence:
            if word.upos is None:
                continue
            counts.setdefault(word.upos, {})
            counts[word.upos].setdefault(word.form, 0)
            counts[word.upos][word.form] += 1

    return normalize(counts)


def invert(frequencies):
    result = {word: {} for words in frequencies.values() for word in words.keys()}

    for pos, words in frequencies.items():
        for word, p in words.items():
            result[word][pos] = p

    return result


class HMM(NamedTuple):
    transitions: Dict[str, Dict[str, float]]
    emissions: Dict[str, Dict[str, float]]
    unknown_emissions: Dict[str, float]


def train_from_conll(training_set, dev_set):
    return HMM(
        transitions=invert(get_transition_frequencies(training_set)),
        emissions=invert(get_emission_frequencies(training_set)),
        unknown_emissions=smoothing(dev_set),
    )


def hmm_ud_english():
    return train_from_conll(ud_treebank("train"), ud_treebank("dev"))
