from typing import List

import pandas as pd

from resources import tokenized_sentences as sentences
from tagger.abc import PosTagger
from tagger.hmm import HMM
from tagger.hmm import hmm_ud_english


def retrace_path(backptr, pos):
    result = [pos]

    for col in reversed(backptr):
        pos = col[pos]
        result.append(pos)

    result.reverse()
    return result


class ViterbiTagger(PosTagger):
    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def _next_col(self, prev_viterbi_col: pd.Series):
        new_viterbi_col = pd.Series()
        new_backptr_col = {}

        for pos in prev_viterbi_col.index:

            path_ps = prev_viterbi_col + self.hmm.transitions.loc[pos]
            backptr = path_ps.idxmax()

            new_backptr_col[pos], new_viterbi_col[pos] = backptr, path_ps[backptr]

        return new_backptr_col, new_viterbi_col

    def pos_tag(self, tokens: List[str]):
        hmm = self.hmm

        # Invece di costruire tutta la matrice, tiene in memoria solo l'ultima colonna.
        viterbi = hmm.transitions["Q0"] + hmm.get_emission(tokens[0])

        # I backptr servono tutti invece.
        backptr = []

        for token in tokens[1:]:
            backptr_col, viterbi = self._next_col(viterbi)
            viterbi += hmm.get_emission(token)
            backptr.append(backptr_col)

        viterbi += hmm.transitions.loc["Qf"]
        path_start = viterbi.idxmax()
        pos_tags = retrace_path(backptr, path_start)

        return list(zip(tokens, pos_tags))


def ud_viterbi_tagger():
    return ViterbiTagger(hmm_ud_english())


if __name__ == "__main__":
    tagger = ud_viterbi_tagger()
    res = tagger.pos_tag(sentences[0])
