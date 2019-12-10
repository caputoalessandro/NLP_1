from functools import reduce
from typing import List

from resources import tokenized_sentences as sentences
from tagger.abc import PosTagger
from tagger.hmm import HMM
from tagger.hmm import hmm_ud_english

from operator import mul
from utils import get_row


def retrace_path(backptr, start):
    return reduce(
        lambda path, backptr_col: [backptr_col[path[0]], *path], reversed(backptr), [start]
    )


def merge_with(fn, d1, d2):
    return {key: fn(d1[key], d2[key]) for key in d1.keys() & d2.keys()}


class ViterbiTagger(PosTagger):
    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def _next_col(self, last_col, token):
        viterbi = {}
        backptr = {}

        for pos in self.hmm.emissions.get(token, self.hmm.unknown_emissions).keys():
            possible_paths_to_pos = merge_with(mul, last_col, self.hmm.transitions[pos])
            viterbi[pos], backptr[pos] = max(
                (v, k) for k, v in possible_paths_to_pos.items()
            )

        return viterbi, backptr

    def pos_tag(self, tokens: List[str]):
        transitions, emissions, default_emissions = self.hmm

        viterbi = merge_with(mul, get_row(transitions, "Q0"), emissions.get(tokens[0], default_emissions))
        backptr = []

        for token in tokens[1:]:
            viterbi, next_backptr = self._next_col(viterbi, token)
            viterbi = merge_with(mul, viterbi, emissions.get(token, default_emissions))
            backptr.append(next_backptr)

        viterbi = merge_with(mul, viterbi, transitions["Qf"])
        path_start = max(viterbi.keys(), key=lambda k: viterbi[k])
        pos_tags = retrace_path(backptr, path_start)

        return list(zip(tokens, pos_tags))


def ud_viterbi_tagger():
    return ViterbiTagger(hmm_ud_english())


if __name__ == "__main__":
    tagger = ud_viterbi_tagger()
    res = tagger.pos_tag(sentences[0])
