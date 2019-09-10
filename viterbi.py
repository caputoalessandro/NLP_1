import pyconll.load
from  training import HMM

UD_ENGLISH_TRAIN = "./resources/en_partut-ud-train.conllu"
training_set = pyconll.load_from_file(UD_ENGLISH_TRAIN)

hmm = HMM.train(training_set)

viterbi = {}

sentences = [
    "The black droid then lowers Vader's mask and helmet onto his head.",
    "These are not the droids your looking for.",
    "Your friends may escape, but you are doomed.",
]

def viterbi(hmm : HMM, sentence : str):
    emissions = hmm.emission
    transitions = hmm.transition
    tokens = sentence.split(" ")
    ris = []
    matrix = {}

    for word in emissions:
        if tokens[0] in word:
            for pos, val in emissions[word].items():
                transition = transitions["Q0"][pos]
                emission = val
                matrix.setdefault(word, {})
                matrix[word].setdefault(pos, 0)
                matrix[word][pos] = emission * transition
                previus_word = tokens[0]

    for word in tokens[1:-2]:
        for pos in emissions:
            if word in pos:
                for previus_pos in viterbi[previus_word]:
                    emission = pos[word]
                    transition = transitions[previus_pos][pos]
                    ris.append(emission * transition * viterbi[previus_word][previus_pos])
                viterbi[word][pos] = max(ris)
        previus_word = word

    for pos in emissions:
        if tokens[-1] in pos:
            emission = pos[tokens[-1]]
            viterbi[tokens[-1]][pos] = emission * transition
            previus_word = tokens[-1]

    return viterbi.values()

if __name__ == "__main__":
    viterbi(hmm, sentences[0])