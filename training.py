from dataclasses import dataclass
from typing import Dict
from toolz.dicttoolz import assoc_in

import pyconll.load

__all__ = ["HMM", "hmm_ud_english"]


def normalize(counts: dict):
    result = {}
    for outer_key, inner_dict in counts.items():
        denom = sum(inner_dict.values())
        result[outer_key] = {
            key: value / denom for key, value in inner_dict.items()
        }
    return result


def get_transition_frequencies(training_set):
    """
    Restituisce un dizionario che contiene i conteggi delle transizioni da
    un elemento a un altro.
    """
    counts = {"Q0": {}, "Qf": {}}

    for sentence in training_set:
        counts["Q0"].setdefault(sentence[0].upos, 0)
        counts["Q0"][sentence[0].upos] += 1
        for t1, t2 in zip(sentence, sentence[1:]):
            if None in (t1.upos, t2.upos):
                continue
            counts.setdefault(t1.upos, {})
            counts[t1.upos].setdefault(t2.upos, 0)
            counts[t1.upos][t2.upos] += 1
        counts["Qf"].setdefault(sentence[-1].upos, 0)
        counts["Qf"][sentence[-1].upos] += 1

    return normalize(counts)


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
    result = {
        word: {} for words in frequencies.values() for word in words.keys()
    }

    for pos, words in frequencies.items():
        for word, p in words.items():
            result[word][pos] = p

    return result


@dataclass
class HMM:
    transition: Dict[str, Dict[str, float]]
    emission: Dict[str, Dict[str, float]]

    @classmethod
    def train(cls, training_set):
        return cls(
            transition=get_transition_frequencies(training_set),
            emission=invert(get_emission_frequencies(training_set)),
        )


def hmm_ud_english():
    training_set = pyconll.load_from_file(
        "resources/en_partut-ud-train.conllu"
    )
    return HMM.train(training_set)


if __name__ == "__main__":
    import json
    from dataclasses import asdict
    from sys import stdout

    def main():
        hmm = hmm_ud_english()
        json.dump(asdict(hmm), stdout, indent=4)

    main()
