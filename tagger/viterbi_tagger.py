from typing import List

import pandas as pd

from resources import tokenized_sentences as sentences
from tagger.abc import PosTagger
from tagger.hmm import HMM
from tagger.hmm import hmm_ud_english
from pprint import pprint


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

    def _find_best_path(self, viterbi: pd.DataFrame, cell_pos: str):
        path_ps = self.hmm.transitions.loc[cell_pos] * viterbi.iloc[:, -1]
        backptr = path_ps.idxmax()
        return backptr, path_ps[backptr]

    def _find_best_paths(self, viterbi: pd.DataFrame):
        new_backptr_col = pd.Series(index=viterbi.index)
        new_viterbi_col = pd.Series(index=viterbi.index)

        for pos in viterbi.index:
            new_backptr_col[pos], new_viterbi_col[pos] = self._find_best_path(
                viterbi, pos
            )

        return new_backptr_col, new_viterbi_col

    def pos_tag(self, tokens: List[str]):
        transitions, emissions, default_emissions = self.hmm

        viterbi = pd.DataFrame()
        backptr = pd.DataFrame()

        try:
            tok_emissions = emissions.loc[tokens[0]]
        except KeyError:
            tok_emissions = default_emissions

        viterbi[0] = transitions["Q0"] * tok_emissions

        for i in range(1, len(tokens)):
            backptr[i], viterbi[i] = self._find_best_paths(viterbi)
            try:
                viterbi[i] *= emissions.loc[tokens[i]]
            except KeyError:
                viterbi[i] *= default_emissions

        path_start = viterbi.iloc[:, -1].idxmax()
        pos_tags = retrace_path(backptr, path_start)

        return list(zip(tokens, pos_tags))


def ud_viterbi_tagger():
    return ViterbiTagger(hmm_ud_english())


if __name__ == "__main__":
    tagger = ud_viterbi_tagger()
    res = tagger.pos_tag(sentences[0])
