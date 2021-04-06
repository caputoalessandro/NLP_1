from resources import Corpus
from tagger.abc import PosTagger
import matplotlib.pyplot as plt
import numpy as np


def correct_tags_count_in_sentence(tagger: PosTagger, sentence):
    golden = [
        (word.form, word.upos) for word in sentence if not word.is_multiword()
    ]
    tokens = [token for token, _ in golden]
    hypothesis = tagger.tagged_tokens(tokens)

    assert len(golden) == len(hypothesis)
    return len(tokens), sum(g == h for g, h in zip(golden, hypothesis))


def correct_tags_ratio_in_corpus(tagger: PosTagger, corpus: Corpus):
    total_tags = 0
    correct_tags = 0

    for sentence in corpus.test:
        sentence_tags, correct_sent_tags = correct_tags_count_in_sentence(
            tagger, sentence
        )
        total_tags += sentence_tags
        correct_tags += correct_sent_tags

    return correct_tags / total_tags


def test_performance(corpus: Corpus, taggers: list[tuple[str, PosTagger]]):
    performance = []
    for name, tagger in taggers:
        performance.append(correct_tags_ratio_in_corpus(tagger, corpus))
        print(f"{name}: {performance[-1]:.4%}")

    labels = ["BASELINE", "NOUN", "NOUN|VERB", "UNIFORM", "STATS"]
    data = [x*100 for x in performance]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects = ax.bar(x, data, width)

    if corpus.name == "la_llct":
        ax.set(ylim=[90, 100])
    else:
        ax.set(ylim=[50, 100])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Ratio')
    ax.set_title('Correct tag ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.bar_label(rects, padding=5)
    plt.show()


def main():
    from tagger import BaselineTagger, HMMTagger
    from tagger.smoothing import ALWAYS_NOUN, ALWAYS_NOUN_OR_VERB, UNIFORM, probability_of_occurring_once

    for corpus in (Corpus.latin(), Corpus.greek()):
        hmm = HMMTagger.train(corpus)
        taggers = [
            ('Baseline', BaselineTagger.train(corpus)),
            ('HMM with NOUN', hmm.with_unknown_emissions(ALWAYS_NOUN)),
            ('HMM with NOUN|VERB', hmm.with_unknown_emissions(ALWAYS_NOUN_OR_VERB)),
            ('HMM with uniform', hmm.with_unknown_emissions(UNIFORM)),
            ('HMM with stats', hmm.with_unknown_emissions(probability_of_occurring_once(corpus)))
        ]

        print(f"\nCorpus {corpus.name}")
        test_performance(corpus, taggers)


if __name__ == "__main__":
    main()
