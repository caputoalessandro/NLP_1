from typing import List

from resources import tokenized_sentences as sentences
from tagger.abc import PosTagger
from tagger.hmm import HMM
from tagger.hmm import hmm_ud_english
from pprint import pprint


class ViterbiTagger(PosTagger):
    def __init__(self, hmm: HMM):
        self.hmm = hmm

    def pos_tag(self, tokens: List[str]):

        transitions, emissions, default_emissions = self.hmm
        viterbi_to_add = {}
        path_to_add = {}
        values = []

        # prima parola
        viterbi_matrix = [
            {
                pos: transitions["Q0"][pos] + em_value
                for pos, em_value in emissions.get(
                    tokens[0], default_emissions
                ).items()
            }
        ]

        backpointer = [{max(viterbi_matrix[0].keys()): "Q0"}]

        # parole centrali
        for token in tokens[1:]:

            for pos, em_value in emissions.get(
                token, default_emissions
            ).items():

                for previus_pos, previus_value in viterbi_matrix[-1].items():

                    values.append(
                        (
                            previus_pos,
                            em_value
                            + transitions.get(previus_pos, {}).get(pos, 0)
                            + previus_value,
                        )
                    )

                previus_pos, value = max(values)
                viterbi_to_add[pos] = value
                path_to_add[pos] = previus_pos
                values = []

            viterbi_matrix.append(viterbi_to_add)
            backpointer.append(path_to_add)
            viterbi_to_add = {}
            path_to_add = {}

        # ultima parola
        previus_pos, value = max(
            (
                previus_pos,
                previus_value + transitions.get(previus_pos, {}).get("Qf", 0),
            )
            for previus_pos, previus_value in viterbi_matrix[-1].items()
        )

        viterbi_matrix.append({pos: value})
        backpointer.append({"Qf": previus_pos})

        rev_backpointer = reversed(backpointer)
        next(rev_backpointer)
        previus_pos = backpointer[-1].get("Qf")
        path = [previus_pos]

        for pointer in rev_backpointer:
            if pointer.get(previus_pos) == "Q0":
                break
            path.append(pointer.get(previus_pos))
            previus_pos = pointer.get(previus_pos)

        return list(zip(tokens, reversed(path)))


def ud_viterbi_tagger():
    return ViterbiTagger(hmm_ud_english())


if __name__ == "__main__":
    tagger = ud_viterbi_tagger()
    tagger.pos_tag(sentences[0])
