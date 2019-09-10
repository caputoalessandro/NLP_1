from training import hmm_ud_english
import pytest


@pytest.fixture(scope="session")
def hmm():
    return hmm_ud_english()
