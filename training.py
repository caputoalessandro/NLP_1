import itertools
from math import log
from dataclasses import dataclass
from typing import Dict


def pairwise(iterator):
    a, b = itertools.tee(iterator)
    next(b, None)
    return zip(a, b)


def get_transition_counts(training_set):
    """
    Restituisce un dizionario che contiene i conteggi delle transizioni da
    un elemento a un altro.
    """
    counts = {"Q0": {}, "Qf": {}}

    for sentence in training_set:
        counts["Q0"].setdefault(sentence[0].upos, 0)
        counts["Q0"][sentence[0].upos] += 1
        for t1, t2 in pairwise(sentence):
            counts.setdefault(t1.upos, {})
            counts[t1.upos].setdefault(t2.upos, 0)
            counts[t1.upos][t2.upos] += 1
        counts["Qf"].setdefault(sentence[-1].upos, 0)
        counts["Qf"][sentence[-1].upos] += 1

    return counts


def get_emission_counts(training_set):
    counts = {}

    for sentence in training_set:
        for word in sentence:
            counts.setdefault(word.form, {})
            counts[word.form].setdefault(word.upos, 0)
            counts[word.form][word.upos] += 1

    return counts


def normalize(counts: dict):
    result = {}
    for outer_key, inner_dict in counts.items():
        denom = log(sum(inner_dict.values()))
        result[outer_key] = {
            key: log(value) - denom for key, value in inner_dict.items()
        }
    return result


@dataclass
class HMM:
    transition: Dict[str, Dict[str, float]]
    emission: Dict[str, Dict[str, float]]

    @classmethod
    def train(cls, training_set):
        return cls(
            transition=normalize(get_transition_counts(training_set)),
            emission=normalize(get_emission_counts(training_set)),
        )


if __name__ == "__main__":
    import pyconll.load

    UD_ENGLISH_TRAIN = "./resources/en_partut-ud-train.conllu"
    training_set = pyconll.load_from_file(UD_ENGLISH_TRAIN)

    hmm = HMM.train(training_set)
    print(hmm)
