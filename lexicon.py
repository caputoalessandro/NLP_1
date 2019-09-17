import yaml
from pprint import pprint


def load_lexicon_en():
    with open("resources/lexicon_en.yaml") as lexicon_file:
        lexicon = yaml.safe_load(lexicon_file)

    pprint(lexicon)


class Lexicon:
    def __init__(self, lemmas):



if __name__ == "__main__":
    load_lexicon_en()
