from collections import Counter

import matplotlib.pyplot as plt
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


def format_errors(errors: Counter):
    return [
        f"{count:>5} {pred} invece di {correct}"
        for (pred, correct), count in errors.most_common(5)
    ]


def plot_accuracies(corpus: Corpus, accuracies: list[float]):
    labels = ["BASELINE", "NOUN", "NOUN|VERB", "UNIFORM", "STATS"]
    data = [x * 100 for x in accuracies]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects = ax.bar(x, data, width)

    if corpus.name == "la_llct":
        ax.set(ylim=[90, 100])
    else:
        ax.set(ylim=[50, 100])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Accuracy")
    ax.set_title("Correct tag ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.bar_label(rects, padding=5)
    plt.show()


def main():
    from tagger import BaselineTagger, HMMTagger
    from tagger.smoothing import (
        ALWAYS_NOUN,
        NOUN_OR_VERB,
        UNIFORM,
        probability_of_occurring_once,
    )

    for corpus in (Corpus.latin(), Corpus.greek()):
        hmm = HMMTagger.train(corpus)
        taggers = [
            BaselineTagger.train(corpus),
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
        tagger_accuracies = [
            (total_tokens - sum(errors.values())) / total_tokens
            for errors in tagger_errors
        ]

        print(f"\n### Corpus {corpus.name} ###\n")
        print(
            tabulate.tabulate(
                zip(tagger_names, tagger_accuracies),
                headers=("Tagger", "Accuracy"),
                floatfmt=".2%",
            )
        )

        print("\nMost common errors\n")

        for tagger_name, errors in zip(tagger_names, tagger_errors):
            print(f"\n{tagger_name}\n")
            total_errors = sum(errors.values())

            print(
                tabulate.tabulate(
                    [
                        (err_count / total_errors, predicted, correct)
                        for (predicted, correct), err_count in errors.most_common(5)
                    ],
                    headers=["Errori", "Predetto", "Corretto"],
                    floatfmt=".2%"
                )
            )

        plot_accuracies(corpus, tagger_accuracies)


if __name__ == "__main__":
    main()
