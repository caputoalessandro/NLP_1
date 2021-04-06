from collections import Counter

from tabulate import tabulate

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


def accuracy(all_errors: Counter, total_tags: int):
    return sum(all_errors.values()) / total_tags


def format_errors(errors: Counter):
    return [
        f".{count:>5} \u2716 {pred} \u2713 {correct}"
        for (pred, correct), count in errors.most_common(5)
    ]


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

        print(f"\nCorpus {corpus.name}\n")
        print(
            tabulate(
                zip(tagger_names, tagger_accuracies),
                headers=("Tagger", "Accuracy"),
                floatfmt=".2%",
            )
        )

        print("\nMost common errors\n")
        print(
            tabulate(
                {
                    name: format_errors(errors)
                    for name, errors in zip(tagger_names, tagger_errors)
                },
                headers="keys",
            )
        )


if __name__ == "__main__":
    main()
