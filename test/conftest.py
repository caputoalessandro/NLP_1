from tagger.hmm import hmm_ud_english
import pytest


@pytest.fixture(scope="session")
def hmm():
    return hmm_ud_english()
