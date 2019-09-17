from collections import defaultdict
from typing import NamedTuple

import yaml

from utils import deepkeys


class Form(NamedTuple):
    lemma: str
    form_key: tuple


def make_form_and_lemma_maps(lemma_file):
    lemmas = yaml.safe_load(lemma_file)
    forms = defaultdict(list)

    for lemma, lemma_data in lemmas.items():
        lemma_forms = lemma_data.get("forms", {"default": lemma})
        for key, form in deepkeys(lemma_forms):
            forms[form].append(Form(lemma, key))

    return forms, lemmas


class Translator:
    def __init__(self):
        with open("resources/lexicon_en.yaml") as lexicon_file:
            self.forms, self.lemmas = make_form_and_lemma_maps(lexicon_file)

    def translate(self, tagged_tokens):
        result = []

        for token in tagged_tokens:
            form = self.forms.get(token.lower())
            if form:
                result.append(form)
            else:
                result.append(token)

        return result


if __name__ == "__main__":
    from sentences import tokenized_sentences
    from pprint import pprint

    t = Translator()
    for sentence in tokenized_sentences:
        res = t.translate(sentence)
        pprint(res)

