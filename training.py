import itertools

import pyconll.load


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
    numerator = {}
    for sentence in training_set:
        for word in sentence:
            numerator.setdefault(word.form, {})
            numerator[word.form].setdefault(word.upos, 0)
            numerator[word.form][word.upos] += 1

    return numerator


def normalize(counts: dict):
    result = {}
    for outer_key, inner_dict in counts.items():
        denom = sum(inner_dict.values())
        result[outer_key] = {
            key: value / denom for key, value in inner_dict.items()
        }
    return result


UD_ENGLISH_TRAIN = "./resources/en_partut-ud-train.conllu"
NGRAM = "Lord of the Rings".split()

train = pyconll.load_from_file(UD_ENGLISH_TRAIN)


if __name__ == "__main__":
    print(train[0][0].upos)
    print(normalize(get_transition_counts(train)))
    print(normalize(get_emission_counts(train)))
    # print(get_emission_freqs(train))
    pass
