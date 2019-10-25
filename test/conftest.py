from tagger import ud_baseline_tagger, ud_viterbi_tagger
import pytest


@pytest.fixture(scope="session")
def baseline_tagger():
    return ud_baseline_tagger()


@pytest.fixture(scope="session")
def viterbi_tagger():
    return ud_viterbi_tagger()
