from tagger.abc import PosTagger
from resources import Corpus
from tagger.smoothing import ALWAYS_NOUN, ALWAYS_NOUN_OR_VERB, UNIFORM


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
    for name, tagger in taggers:
        performance = correct_tags_ratio_in_corpus(tagger, corpus)
        print(f"{name}: {performance:.4%}")


def main():
    from tagger import BaselineTagger, HMMTagger
    from tagger.smoothing import probability_of_occurring_once

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
