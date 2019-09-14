from viterbi import viterbi
from training import hmm_ud_english
import pyconll.load


def performance():

    hmm = hmm_ud_english()

    test_set = pyconll.load_from_file("resources/en_partut-ud-test.conllu")

    errors = 0
    tokens = 0
    for sentence in test_set:
        test = [(pos.form, pos.upos) for pos in sentence]
        pos_tagger = viterbi(hmm, [pos.form for pos in sentence])
        compare = zip(test, pos_tagger)
        tokens = tokens + len(test)
        for t, p in compare:
            if t != p:
                errors += 1
                if p[1] != "PROPN":
                    print(t, p)

    print(tokens)
    print(errors)
    print((errors/tokens) * 100)




if __name__ == "__main__":
    performance()
