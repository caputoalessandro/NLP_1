from tagger.abc import PosTagger
from resources import ud_treebank
from pyconll.unit import Sentence
import logging

logger = logging.getLogger()


def count_correct_tags(tagger: PosTagger, sentence: Sentence):
    golden = [(word.form, word.upos) for word in sentence if not word.is_multiword()]
    tokens = [token for token, _ in golden]
    hypothesis = tagger.tagged_tokens(tokens)

    assert len(golden) == len(hypothesis)
    return len(tokens), sum(g == h for g, h in zip(golden, hypothesis))


def tagger_performance(tagger: PosTagger):
    test_set = ud_treebank("test")

    total_tags = 0
    correct_tags = 0

    for sentence in test_set:
        sentence_tags, correct_sent_tags = count_correct_tags(tagger, sentence)
        total_tags += sentence_tags
        correct_tags += correct_sent_tags

    return correct_tags / total_tags
