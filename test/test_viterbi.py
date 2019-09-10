import pyconll.load
import pytest
from training import HMM
from viterbi import viterbi


sentences = [
    "The black droid then lowers Vader's mask and helmet onto his head.",
    "These are not the droids your looking for.",
    "Your friends may escape, but you are doomed.",
]


UD_ENGLISH_TRAIN = "./resources/en_partut-ud-train.conllu"
training_set = pyconll.load_from_file(UD_ENGLISH_TRAIN)

hmm = HMM.train(training_set)


@pytest.mark.parametrize("sentence", sentences)
def test_viterbi(sentence):
    result = viterbi(hmm, sentence)
    print(result)
