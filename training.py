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


def get_transition_freqs(training_set):
    dicts = get_transition_counts(training_set)
    probabilities = {}
    for t1, t2 in dicts.items():
        denom = sum(t2.values())
        for key in t2:
            numerator = t2[key]
            probabilty = numerator / denom
            probabilities[t1] = t2
            probabilities[t1][key] = probabilty
    return probabilities


def get_emission_freq():
    numerator = {}
    for sentence in train:
         for word in sentence:
            numerator.setdefault(word.form, {})
            numerator[word.form].setdefault(word.upos, 0)
            numerator[word.form][word.upos] += 1

    return numeratorx

UD_ENGLISH_TRAIN = "./resources/en_partut-ud-train.conllu"
NGRAM = "Lord of the Rings".split()

train = pyconll.load_from_file(UD_ENGLISH_TRAIN)


if __name__ == "__main__":
    print(train[0][0].upos)
    print(get_transition_freqs(train))
    # print(get_emission_freqs(train))
    pass
