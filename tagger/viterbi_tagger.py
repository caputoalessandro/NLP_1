from typing import List

import pandas as pd

from resources import tokenized_sentences as sentences
from tagger.abc import PosTagger
from tagger.hmm import HMM
from tagger.hmm import hmm_ud_english


def retrace_path(backptr: pd.DataFrame, pos):
    result = [pos]

    for _, col in reversed(list(backptr.items())):
        pos = col[pos]
        result.append(pos)

    result.reverse()
    return result


class ViterbiTagger(PosTagger):
    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def _best_subpaths(self, prev_viterbi_col: pd.Series):
        new_backptr_col = pd.Series()
        new_viterbi_col = pd.Series()

        for pos in prev_viterbi_col.index:
            path_ps = self.hmm.transitions.loc[pos] * prev_viterbi_col
            backptr = path_ps.idxmax()

            new_backptr_col[pos], new_viterbi_col[pos] = backptr, path_ps[backptr]

        return new_backptr_col, new_viterbi_col

    def pos_tag(self, tokens: List[str]):
        transitions = self.hmm.transitions

        viterbi = pd.DataFrame()
        backptr = pd.DataFrame()

        viterbi[0] = transitions["Q0"] * self._get_emission(tokens[0])

        for i in range(1, len(tokens)):
            backptr[i], viterbi[i] = self._best_subpaths(viterbi.iloc[:, -1])
            viterbi[i] *= self._get_emission(tokens[i])

        viterbi.iloc[:, -1] *= transitions.loc["Qf"]
        path_start = viterbi.iloc[:, -1].idxmax()
        pos_tags = retrace_path(backptr, path_start)

        return list(zip(tokens, pos_tags))

    def _get_emission(self, token):
        try:
            return self.hmm.emissions.loc[token]
        except KeyError:
            return self.hmm.unknown_emissions


def ud_viterbi_tagger():
    return ViterbiTagger(hmm_ud_english())


if __name__ == "__main__":
    tagger = ud_viterbi_tagger()
    res = tagger.pos_tag(sentences[0])
