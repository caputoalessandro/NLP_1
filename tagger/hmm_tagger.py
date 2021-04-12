from toolz.curried import pipe, valmap

from resources import POS_TAGS, Corpus
from tagger.abc import PosTagger
from utils import (
    get_row,
    sum_values,
    DictWithMissing,
    counts_to_log_probability,
    transpose,
)

__all__ = ["HMMTagger"]


def transition_counts(training_set):
    # Smoothing: dai conteggio iniziale 1 a tutte le transizioni
    counts = {pos: dict.fromkeys([*POS_TAGS, "Qf"], 1) for pos in [*POS_TAGS, "Q0"]}

    for sentence in training_set:
        sentence = [word for word in sentence if not word.is_multiword()]

        counts["Q0"][sentence[0].upos] += 1
        for t1, t2 in zip(sentence, sentence[1:]):
            counts[t1.upos][t2.upos] += 1
        counts[sentence[-1].upos]["Qf"] += 1

    return counts


def emission_counts(training_set):
    counts = {}

    for sentence in training_set:
        for word in sentence:
            if word.upos is None:
                continue
            counts.setdefault(word.upos, {}).setdefault(word.form, 0)
            counts[word.upos][word.form] += 1

    return counts


class HMMTagger(PosTagger):
    def __init__(self, transitions, emissions):
        self.transitions = transitions
        self.emissions = emissions

    @classmethod
    def train(cls, corpus: Corpus):
        transitions = pipe(
            corpus.train,
            transition_counts,
            valmap(counts_to_log_probability),
            transpose,
        )
        emissions = pipe(
            corpus.train,
            emission_counts,
            valmap(counts_to_log_probability),
            transpose,
        )
        return cls(transitions, emissions)

    def with_unknown_emissions(self, ue):
        return HMMTagger(
            self.transitions, DictWithMissing(self.emissions).with_missing(ue)
        )

    def pos_tags(self, tokens: list[str]):
        transitions, emissions = self.transitions, self.emissions

        # inizializzazione di viterbi prima colonna
        # effettuiamo la somma (perchè lavoriamo con logaritmi) tra la
        # probabilità di transizione e la probabilità di emissione

        viterbi = [sum_values(get_row(transitions, "Q0"), emissions[tokens[0]])]
        backptr = []

        for token in tokens[1:]:
            next_viterbi, next_backptr = self._next_col(viterbi[-1], token)

            viterbi.append(next_viterbi)
            backptr.append(next_backptr)

        # sommo a transizioni con pseudotoken finale
        viterbi.append(sum_values(viterbi[-1], transitions["Qf"]))

        # calcolo il path partendo dal PoS con probabilità massima
        path = [max(viterbi[-1].keys(), key=lambda k: viterbi[-1][k])]

        for col in reversed(backptr):
            path.insert(0, col[path[0]])

        return path

    def _next_col(self, last_col, token):
        # calcola la prossima colonna di viterbi

        transitions, emissions = self.transitions, self.emissions

        viterbi = {}
        backptr = {}

        for pos in emissions[token].keys():

            # calcolo tutti i percorsi che vanno  dal pos precedente al pos corrente
            paths_to_pos = sum_values(last_col, transitions[pos])

            # tra tutti i path scelgo quello con probabilità massima, inserisco il pos
            # all'interno della colonna e del backpointer
            backptr[pos], viterbi[pos] = max(paths_to_pos.items(), key=lambda it: it[1])

        # sommo emissioni
        viterbi = sum_values(viterbi, emissions[token])
        return viterbi, backptr
