import pytest
from tagger import ud_baseline_tagger, ud_viterbi_tagger
from tagger.evaluation import tagger_performance

def test_performance():
    baseline = ud_baseline_tagger()
    viterbi = ud_viterbi_tagger()

    baseline_performance = tagger_performance(baseline)
    print(f"Baseline: {baseline_performance}")
    viterbi_performance = tagger_performance(viterbi)
    print(f"Viterbi: {viterbi_performance}")


    assert baseline_performance < viterbi_performance
