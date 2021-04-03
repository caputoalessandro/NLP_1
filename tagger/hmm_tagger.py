from operator import add
from typing import List
from tagger.abc import PosTagger
from tagger.hmm import HMM
from utils import get_row, disjunction_apply


def retrace_path(backptr, start):
    path = [start]
    for col in reversed(backptr):
        path.append(col[path[-1]])
    path.reverse()
    return path


class HMMTagger(PosTagger):
    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def _next_col(self, last_col, token):
        transitions, emissions = self.hmm

        viterbi = {}
        backptr = {}

        for pos in emissions[token].keys():
            paths_to_pos = disjunction_apply(add, last_col, transitions[pos])
            backptr[pos], viterbi[pos] = max(
                paths_to_pos.items(), key=lambda it: it[1]
            )

        viterbi = disjunction_apply(add, viterbi, emissions[token])
        return viterbi, backptr

    def pos_tags(self, tokens: List[str]):
        transitions, emissions = self.hmm

        # Mantiene in memoria solo l'ultima colonna invece di tutta la matrice.
        viterbi = disjunction_apply(
            add, get_row(transitions, "Q0"), emissions[tokens[0]]
        )
        backptr = []

        for token in tokens[1:]:
            viterbi, next_backptr = self._next_col(viterbi, token)
            backptr.append(next_backptr)

        viterbi = disjunction_apply(add, viterbi, transitions["Qf"])
        path_start = max(viterbi.keys(), key=lambda k: viterbi[k])
        return retrace_path(backptr, path_start)

    def with_unknown_emissions(self, ue):
        return HMMTagger(self.hmm.with_unknown_emissions(ue))

    @classmethod
    def train(cls, *args, **kwargs):
        return cls(HMM.train(*args, **kwargs))
