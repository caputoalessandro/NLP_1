from pprint import pprint
from typing import List

from training import HMM


def make_default_emissions(pos_list):

    vit = {pos: 1 / len(pos_list) for pos in pos_list}

    vit["PROPN"] = 1

    return vit


def viterbi(hmm: HMM, tokens: List[str]):

    default_emissions = make_default_emissions(hmm.transition.keys())
    matrix = {}
    emissions = hmm.emission
    transitions = hmm.transition
    # pprint(emissions)

    for pos, val in emissions.get(tokens[0], default_emissions).items():
        print(default_emissions)
        print("primo caso")
        print(tokens[0])
        emission = val
        transition = transitions['Q0'][pos]
        matrix.setdefault(tokens[0], {})
        matrix[tokens[0]][pos] = emission * transition
    backpointer = tokens[0]

    for token in tokens[1:-1]:
        print("caso generale")
        for pos, val in emissions.get(token, default_emissions).items():
            print("cosa  strana", emissions.get(token, default_emissions).items() )
            max = -1000
            max_pos = "NOMAX?"
            for last_pos in matrix.get(backpointer):
                emission = val
                last_val = matrix[backpointer][last_pos]
                print("pos",token,pos)
                print("last_pos",backpointer,last_pos)
                transition = transitions[pos][last_pos]
                ris = emission * transition * last_val
                if max < ris:
                    max = ris
                    max_pos = pos
            matrix.setdefault(token, {})
            matrix[token][max_pos] = max
        backpointer = token

    max = -1000
    for last_ris in matrix.get(backpointer):
        print("caso finale")
        transition = transitions['Qf'][max_pos]
        ris = transition * last_ris
        if max < ris:
            max = ris
    matrix.setdefault(token, {})
    matrix[token]["Qf"] = max

    print(matrix)


if __name__ == "__main__":
    from training import hmm_ud_english
    from sentences import tokenized_sentences as sentences

    hmm = hmm_ud_english()
    viterbi(hmm, sentences[0])

# else:
#     if pos is "PROPN":
#         print("we")
#         emission = 1
#         transation = transations["Q0"][pos]
#         matrix.setdefault(pos, {})
#         matrix[pos][token] = emission * transation
#     else:
#         emission = 1 / emissions.__len__()
#         transation = transations["Q0"][pos]
#         matrix.setdefault(pos, {})
#         matrix[pos][token] = emission * transation
