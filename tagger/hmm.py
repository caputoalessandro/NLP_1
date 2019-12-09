from typing import NamedTuple

import numpy as np
import pandas as pd

from resources import ud_treebank


def transition_counts(training_set):
    """
    Restituisce un dizionario che contiene i conteggi delle transizioni da
    un elemento a un altro.
    """
    counts = {"Q0": {}}

    for sentence in training_set:
        sentence = [word for word in sentence if not word.is_multiword()]

        counts["Q0"].setdefault(sentence[0].upos, 0)
        counts["Q0"][sentence[0].upos] += 1
        for t1, t2 in zip(sentence, sentence[1:]):
            counts.setdefault(t1.upos, {})
            counts[t1.upos].setdefault(t2.upos, 0)
            counts[t1.upos][t2.upos] += 1
        counts[sentence[-1].upos].setdefault("Qf", 0)
        counts[sentence[-1].upos]["Qf"] += 1

    return counts


def emission_counts(training_set):
    counts = {}

    for sentence in training_set:
        for word in sentence:
            if word.upos is None:
                continue
            counts.setdefault(word.upos, {})
            counts[word.upos].setdefault(word.form, 0)
            counts[word.upos][word.form] += 1

    return counts


def smoothing_counts(development_set):
    smoothing_dict = {}
    count_dict = {}

    # conto occorrenze parole
    for sentence in development_set:
        for word in sentence:
            # if not word.is_multiword():
            count_dict.setdefault(word.form, 0)
            count_dict[word.form] += 1

    # conto quante volte occorre un pos solo per le parole cche appaiono una volta
    for sentence in development_set:
        for word in sentence:
            if count_dict[word.form] == 1 and not word.is_multiword():
                smoothing_dict.setdefault(word.upos, 0)
                smoothing_dict[word.upos] += 1

    return smoothing_dict


class HMM(NamedTuple):
    transitions: pd.DataFrame
    emissions: pd.DataFrame
    unknown_emissions: pd.Series

    def get_emission(self, token):
        try:
            return self.emissions.loc[token]
        except KeyError:
            return self.unknown_emissions


def divide_by_total_log(s: pd.Series):
    return s.apply(np.log) - np.log(s.sum())


def counts_to_probs_df(counts: dict):
    df = pd.DataFrame.from_dict(counts).fillna(0)
    df = df.apply(divide_by_total_log)
    return df


def counts_to_probs_series(counts: dict):
    return divide_by_total_log(pd.Series(counts))


def train_from_conll(training_set, dev_set):
    transitions = counts_to_probs_df(transition_counts(training_set))
    emissions = counts_to_probs_df(emission_counts(training_set))
    smoothing = counts_to_probs_series(smoothing_counts(dev_set))

    return HMM(transitions, emissions, smoothing)


def hmm_ud_english():
    return train_from_conll(ud_treebank("train"), ud_treebank("dev"))
