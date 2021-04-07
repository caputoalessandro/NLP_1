from collections import Counter
from math import log

from resources import Corpus, POS_TAGS
from utils import counts_to_log_probability

ALWAYS_NOUN = {
    'NOUN': log(1)
}

NOUN_OR_VERB = {
    'NOUN': log(.5),
    'VERB': log(.5)
}

UNIFORM = counts_to_log_probability(dict.fromkeys(POS_TAGS, 1))


def probability_of_occurring_once(corpus: Corpus):
    word_to_pos = {}

    for sentence in corpus.dev:
        for word in sentence:
            if word.form in word_to_pos:
                word_to_pos[word.form] = None
            else:
                word_to_pos[word.form] = word.upos

    return counts_to_log_probability(Counter(pos for pos in word_to_pos.values() if pos is not None))
