from tagger import ud_baseline_tagger, ud_viterbi_tagger
from translate.translator import DirectTranslator
import pytest


@pytest.fixture(scope="session")
def baseline_tagger():
    return ud_baseline_tagger()


@pytest.fixture(scope="session")
def viterbi_tagger():
    return ud_viterbi_tagger()


@pytest.fixture(scope="session")
def viterbi_translator(viterbi_tagger):
    return DirectTranslator(viterbi_tagger)
