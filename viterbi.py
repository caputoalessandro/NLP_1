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
    saved_path = []
    # print(tokens)
    # print("ewweeeee")
    # pprint(transitions)

    for pos, val in emissions.get(tokens[0], default_emissions).items():

        emission = val
        if transitions["Q0"].get(pos):
            transition = transitions['Q0'][pos]
        else:
            transition = 1 / len(transitions.keys())
        matrix.setdefault(tokens[0], {})
        matrix[tokens[0]][pos] = emission * transition
        backpointer = tokens[0]

    saved_path.append((tokens[0], "Qi"))

    for token in tokens[1:]:

        if emissions.get(token):

            for pos, val in emissions.get(token).items():
                max = -1000
                max_pos = "NOMAX?"
                max_path_val = -1000

                for last_pos in matrix.get(backpointer):
                    emission = val
                    last_val = matrix[backpointer][last_pos]

                    if transitions[last_pos].get(pos):
                        transition = transitions[last_pos][pos]
                        ris = emission * transition * last_val
                    else:
                        transition = 1 / len(transitions.keys())
                        ris = emission * transition * last_val

                    saved_path_val = last_val * transition

                    if max < ris:
                        max = ris


                    if max_path_val < saved_path_val:
                        max_path_val = saved_path_val
                        max_pos = last_pos

                matrix.setdefault(token, {})
                matrix[token][pos] = max

        else:
            max = -1000
            max_pos = "NOMAX?"
            max_path_val = -1000

            for last_pos in matrix.get(backpointer):
                emission = 1
                # print("pos", pos)
                # print("lastpos", last_pos)
                last_val = matrix[backpointer][last_pos]
                transition = 1 / len(transitions.keys())
                ris = emission * transition * last_val
                if max < ris:
                    max = ris

                saved_path_val = last_val * transition
                if max_path_val < saved_path_val:
                    max_path_val = saved_path_val
                    max_pos = last_pos

            matrix.setdefault(token, {})
            matrix[token]["PROPN"] = max

        backpointer = token

        saved_path.append((token, max_pos))

    max = -1000
    for last_pos in matrix.get(backpointer):
        # print("caso finale")
        # print(last_pos)
        transition = transitions[last_pos]["Qf"]

        last_val = matrix[backpointer][last_pos]
        ris = transition * last_val
        if max < ris:
            max = ris
            max_pos = last_pos

    saved_path.append(("Qf", max_pos))
    matrix.setdefault("Qf", {})
    matrix["Qf"]["Qf"] = max

    # pprint(matrix)

    i = len(tokens)
    path = []
    for token in reversed(tokens):
        path.append((token, saved_path[i][1]))
        i -= 1

    path.reverse()
    return path

if __name__ == "__main__":
    from training import hmm_ud_english
    from sentences import tokenized_sentences as sentences

    hmm = hmm_ud_english()
    viterbi(hmm, sentences[1])
