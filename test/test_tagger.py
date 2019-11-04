import pytest
from pprint import pprint
from tagger.evaluation import tagger_performance
from resources import tokenized_sentences


def test_performance(baseline_tagger, viterbi_tagger):

    baseline_performance = tagger_performance(baseline_tagger)
    print(f"Baseline: {baseline_performance:.2%}")
    viterbi_performance = tagger_performance(viterbi_tagger)
    print(f"Viterbi: {viterbi_performance:.2%}")

    assert baseline_performance < viterbi_performance


@pytest.mark.parametrize("tokens", tokenized_sentences)
def test_sentences(tokens, viterbi_tagger):
    tagged_tokens = viterbi_tagger.pos_tag(tokens)
    pprint(tagged_tokens)
