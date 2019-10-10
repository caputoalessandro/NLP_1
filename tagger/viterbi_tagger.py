from typing import List
from tagger.abc import PosTagger
from tagger.hmm import HMM
from pprint import pprint
from tagger.smoothing import smoothing
from tagger.hmm import hmm_ud_english
from sentences import tokenized_sentences as sentences


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
        default_emissions = smoothing()
        dict_to_add = {}
        backpointer = []

        # prima parola
        viterbi_matrix = [
            {
                pos: transitions["Q0"][pos] * em_value
                for pos, em_value in emissions.get(
                    tokens[0], default_emissions
                ).items()
            }
        ]

        # add_to_path = [
        #     (
        #         pos,
        #         1.0
        #         * transitions.get("Q0", {}).get(pos, 0)
        #     )
        #     for pos, em_value in emissions.get(
        #         tokens[0], default_emissions
        #     ).items()
        # ]

        # pos_to_add = max(add_to_path, key=lambda x: x[1])
        # tuple_to_add = (tokens[0], pos_to_add[0])
        # backpointer.append("start")

        # parole centrali
        for token in tokens[1:]:

            for pos, em_value in emissions.get(
                token, default_emissions
            ).items():

                to_add = [
                    (
                        pos,
                        em_value
                        * transitions.get(previus_pos, {}).get(pos, 0)
                        * previus_value,
                    )
                    for previus_pos, previus_value in viterbi_matrix[
                        -1
                    ].items()
                ]

                value_to_add = max(to_add, key=lambda x: x[1])
                dict_to_add[value_to_add[0]] = value_to_add[1]

                add_to_path = [
                    (
                        previus_pos,
                        previus_value
                        * transitions.get(previus_pos, {}).get(pos, 0),
                    )
                    for previus_pos, previus_value in viterbi_matrix[
                        -1
                    ].items()
                ]

            viterbi_matrix.append(dict_to_add)
            pos_to_add = max(add_to_path, key=lambda x: x[1])
            # tuple_to_add = (token, pos_to_add[0])
            backpointer.append(pos_to_add[0])
            dict_to_add = {}

        # ultima parola
        to_add = [
            (
                pos,
                previus_value * transitions.get(previus_pos, {}).get("Qf", 0),
            )
            for previus_pos, previus_value in viterbi_matrix[-1].items()
        ]

        value_to_add = max(to_add, key=lambda x: x[1])
        dict_to_add[value_to_add[0]] = value_to_add[1]
        viterbi_matrix.append(to_add)
        # print(viterbi_matrix)

        add_to_path = [
            (
                previus_pos,
                previus_value * transitions.get(previus_pos, {}).get("Qf", 0),
            )
            for previus_pos, previus_value in viterbi_matrix[-1]
        ]

        pos_to_add = max(add_to_path, key=lambda x: x[1])
        # tuple_to_add = (tokens[-1], pos_to_add[0])
        # print(pos_to_add)
        backpointer.append(pos_to_add[0])
        res = list(zip(tokens,backpointer))
        print(res)
        # print(backpointer)
        # print(tokens)
        # pprint(viterbi_matrix)
        return res


def ud_viterbi_tagger():
    return ViterbiTagger(hmm_ud_english())


if __name__ == "__main__":

    tagger = ud_viterbi_tagger()
    tagger.pos_tag(sentences[1])
