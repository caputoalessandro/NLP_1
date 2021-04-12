from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import tabulate

from resources import Corpus
from tagger.abc import PosTagger


def sentence_errors(tagger: PosTagger, golden):
    errors = Counter()
    tokens = [token for token, _ in golden]
    hypothesis = tagger.pos_tags(tokens)

    assert len(golden) == len(hypothesis)

    for (_, correct), prediction in zip(golden, hypothesis):
        if prediction != correct:
            errors[(prediction, correct)] += 1

    return errors


def corpus_errors(tagger: PosTagger, corpus: Corpus):
    errors = Counter()

    for sentence in corpus.test:
        tagged_tokens = [
            (word.form, word.upos) for word in sentence if not word.is_multiword()
        ]
        errors += sentence_errors(tagger, tagged_tokens)

    return errors


def total_test_tokens(corpus):
    return sum(
        1 for sentence in corpus.test for word in sentence if not word.is_multiword()
    )


def plot_accuracies(accuracies: dict[str, list[float]]):
    labels = [
        "BASELINE",
        "NOUN",
        "N | V",
        "UNIFORM",
        "STATS",
    ]
    x = np.arange(len(labels))
    width = 0.35

    cm = 1 / 2.54
    fig, axs = plt.subplots(1, 2, figsize=(22 * cm, 10 * cm))

    for ax, (corpus_name, corpus_accuracies) in zip(axs, accuracies.items()):

        sorted_acc_and_labels = sorted(zip(corpus_accuracies, labels))
        sorted_acc, sorted_labels = [
            [x[0] for x in sorted_acc_and_labels],
            [x[1] for x in sorted_acc_and_labels],
        ]

        data = [x * 100 for x in sorted_acc]
        rects = ax.bar(x, data, width)

        if corpus_name == "la_llct":
            ax.set(ylim=[90, 100])
        else:
            ax.set(ylim=[50, 100])

        if corpus_name == "la_llct":
            ax.set_ylabel("Accuracy")

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_title(corpus_name)
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_labels)

        ax.bar_label(rects, fmt="%.2f", padding=5)

    plt.show()


def main():
    from tagger import BaselineTagger, HMMTagger
    from tagger.smoothing import (
        ALWAYS_NOUN,
        NOUN_OR_VERB,
        UNIFORM,
        probability_of_occurring_once,
    )

    accuracies = {}

    for corpus in (Corpus.latin(), Corpus.greek()):
        hmm = HMMTagger.train(corpus)
        baseline = BaselineTagger.train(corpus)
        taggers = [
            baseline.with_default_for_missing("NOUN"),
            hmm.with_unknown_emissions(ALWAYS_NOUN),
            hmm.with_unknown_emissions(NOUN_OR_VERB),
            hmm.with_unknown_emissions(UNIFORM),
            hmm.with_unknown_emissions(probability_of_occurring_once(corpus)),
        ]

        tagger_names = [
            "Baseline",
            "HMM: Always NOUN",
            "HMM: 0.5 NOUN, 0.5 VERB",
            "HMM: 1/#PosTags",
            "HMM: Stats on occurring once",
        ]

        tagger_errors = [corpus_errors(tagger, corpus) for tagger in taggers]
        total_tokens = total_test_tokens(corpus)
        accuracies[corpus.name] = [
            (total_tokens - sum(errors.values())) / total_tokens
            for errors in tagger_errors
        ]

        print(f"\n### Corpus {corpus.name} ###\n")
        print(
            tabulate.tabulate(
                sorted(
                    zip(tagger_names, accuracies[corpus.name]),
                    key=lambda it: it[1],
                    reverse=True,
                ),
                headers=("Tagger", "Accuracy"),
                floatfmt=".2%",
            )
        )

        print("\nMost common errors\n")

        for tagger_name, errors in zip(tagger_names, tagger_errors):
            total_errors = sum(errors.values())

            print(f"\n{tagger_name}\n")
            print(
                tabulate.tabulate(
                    [
                        (err_count / total_errors, correct, predicted)
                        for (predicted, correct), err_count in errors.most_common(5)
                    ],
                    headers=["Errori", "Corretto", "Predetto"],
                    floatfmt=".2%",
                )
            )

    plot_accuracies(accuracies)


if __name__ == "__main__":
    main()
