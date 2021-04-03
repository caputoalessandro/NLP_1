from tagger.abc import PosTagger
from resources import Corpus


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


def test_performance(corpus: Corpus, taggers: list[PosTagger]):
    for tagger in taggers:
        performance = correct_tags_ratio_in_corpus(tagger, corpus)
        tagger_name = type(tagger).__name__
        print(f"{tagger_name}: {performance:.4%}")


def main():
    from tagger import BaselineTagger, HMMTagger

    corpus = Corpus.latin()
    baseline = BaselineTagger.train(corpus)
    viterbi = HMMTagger.train(corpus)

    test_performance(corpus, [baseline, viterbi])


if __name__ == "__main__":
    main()
