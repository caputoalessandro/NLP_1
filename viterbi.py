import pyconll.load
import numpy as np
from training import HMM
from pprint import pprint
from sentences import tokenized_sentences


UD_ENGLISH_TRAIN = "./resources/en_partut-ud-train.conllu"
training_set = pyconll.load_from_file(UD_ENGLISH_TRAIN)

hmm = HMM.train(training_set)

sentences = [
    "The black droid then lowers Vader's mask and helmet onto his head"
    "These are not the droids your looking for"
    "Your friends may escape, but you are doomed"
]


matrix = {}

def viterbi(hmm: HMM, sen: str):
    # inizializzo la matrice
    emissions = hmm.emission
    transations = hmm.transition
    tokens = sen.split(" ")
    for word in tokens:
        if word in emissions:
            print(transations)
            pos = emissions.get(word)
            print("we", pos)
            for p in pos:
                emission = emissions[word][p]
                transation = transations["Q0"][p]
                matrix.setdefault(word, {})
                matrix[word][p] = emission * transation
    pprint(matrix)


if __name__ == "__main__":
    viterbi(hmm, sentences[0])

#
#
# emissions = hmm.emission
#   transitions = hmm.transition
#   tokens = sentence.split(" ")
#   ris = []
#   matrix = {}
#
#   # for word in emissions:
#   #     if tokens[0] in word:
#   word = tokens[0]
#   if word in emissions:
#       for pos, val in emissions[word].items():
#           transition = transitions["Q0"][pos]
#           emission = val
#           matrix.setdefault(word, {})
#           matrix[word].setdefault(pos, 0)
#           matrix[word][pos] = emission * transition
#           previus_word = word
#           #print(matrix)
#   else if
#   for word in tokens[1:-2]:
#       # for word in emissions:
#       #     if token in word:
#       max_val = 0
#       max_pos = " "
#       if word in emissions:
#           for pos, val in emissions[word].items():
#               for previus_pos in matrix[previus_word]:
#                   emission = val
#                   transition = transitions[previus_pos][pos]
#                   ris_val = emission * transition * matrix[previus_word][previus_pos]
#                   ris.append(ris_val)
#                   if ris_val > 0: max_pos = pos, max_value = ris_val
#           matrix.setdefault(word, {})
#           matrix[word].setdefault(max_pos, 0)
#           matrix[word][max_pos] = max_val
#           previus_word = word
#           print(matrix)
#
#   for pos in emissions:
#       if tokens[-1] in pos:
#           emission = pos[tokens[-1]]
#           matrix[tokens[-1]][pos] = emission * transition
#           previus_word = tokens[-1]
#
#   return matrix.values()
