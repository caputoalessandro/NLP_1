import pytest

from viterbi import viterbi
from sentences import tokenized_sentences


@pytest.mark.parametrize("tokens", tokenized_sentences)
def test_viterbi(hmm, tokens):
    result = viterbi(hmm, tokens)
    print(result)