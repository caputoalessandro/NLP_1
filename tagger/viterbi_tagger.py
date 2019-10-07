from typing import List
from tagger.abc import PosTagger
from tagger.hmm import HMM
from pprint import pprint


def make_default_emissions(pos_list):
    # vit = {pos: 1 / len(pos_list) for pos in pos_list}
    vit = {}
    vit["PROPN"] = 1
    return vit


class ViterbiTagger(PosTagger):
    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def pos_tag(self, tokens: List[str]):
        transitions, emissions = self.hmm
        default_emissions = make_default_emissions(transitions.keys())
        dict_to_add = {}
        # pprint(transitions)
        # pprint(emissions)

        # prima parole
        viterbi_matrix = [
            {
                pos: em_value * transitions["Q0"][pos]
                for pos, em_value in emissions.get(
                    tokens[0], default_emissions
                ).items()
            }
        ]


        # parole centrali
        for token in tokens[1:-1]:

            previus_column = len(viterbi_matrix) - 1

            for pos, em_value in emissions.get(
                token, default_emissions
            ).items():

                to_add = [
                    (
                        pos,
                        em_value
                        * transitions[previus_pos][pos]
                        * previus_value
                    )
                    for previus_pos, previus_value in viterbi_matrix[
                        previus_column
                    ].items()
                ]

                # print("to_add", to_add)

                value_to_add = max(to_add, key=lambda x: x[1])

                # print("value  to add ", value_to_add)

                dict_to_add[value_to_add[0]] = value_to_add[1]

            viterbi_matrix.append(dict_to_add)

            dict_to_add = {}

        # ultima parola
        to_add = {
            pos: em_value * transitions[pos]["Qf"]
            for pos, em_value in emissions.get(
                tokens[-1], default_emissions
            ).items()
        }

        viterbi_matrix.append(to_add)

        pprint(viterbi_matrix)


def ud_viterbi_tagger():
    return ViterbiTagger(hmm_ud_english())


if __name__ == "__main__":
    from tagger.hmm import hmm_ud_english
    from sentences import tokenized_sentences as sentences

    tagger = ud_viterbi_tagger()
    tagger.pos_tag(sentences[1])
