from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict
from toolz import get_in

from yaml import FullLoader, load

from utils import deepkeys


@dataclass
class Lemma:
    id: str
    pos: str
    forms: dict = field(default_factory=dict)


@dataclass
class Form:
    lemma: str
    form_key: tuple


def populate_defaults_en(lemmas: Dict[str, Lemma]):
    for lemma in lemmas.values():
        lemma.forms["default"] = lemma.id
        if lemma.pos == "NOUN":
            lemma.forms["fem"] = lemma.id


def make_lemma_map(lemma_file):
    lemmas = load(lemma_file, FullLoader)
    lemmas = {
        lemma_id: Lemma(lemma_id, **lemma_spec)
        for lemma_id, lemma_spec in lemmas.items()
    }
    return lemmas


def make_form_and_lemma_maps_en(lemma_file):

    lemmas = make_lemma_map(lemma_file)
    populate_defaults_en(lemmas)

    forms = defaultdict(list)

    for lemma_id, lemma in lemmas.items():
        for key, form in deepkeys(lemma.forms):
            forms[form].append(Form(lemmas[lemma_id], key))

    return forms, lemmas


MULTIWORDS = {("'", "re"): "are", ("'", "s"): "__genitive__"}


def map_multiwords(tokens: List[str]):
    pair_iter = zip(tokens, tokens[1:])
    result = []

    for pair in pair_iter:
        if pair in MULTIWORDS:
            result.append(MULTIWORDS[pair])
            next(pair_iter)
        else:
            result.append(pair[0])

    return result


class Translator:
    def __init__(self):
        with open("resources/lexicon_en.yaml") as lexicon_file:
            self.forms, self.lemmas_en = make_form_and_lemma_maps_en(
                lexicon_file
            )
        with open("resources/lexicon_it.yaml") as lexicon_file:
            self.lemmas_it = make_lemma_map(lexicon_file)

        with open("resources/en_to_it.yaml") as lexicon_file:
            self.en_to_it = load(lexicon_file, FullLoader)

    def translate(self, tokens):
        result = []
        tokens = map_multiwords(tokens)

        for token in tokens:
            forms = self.forms.get(token.lower())
            if forms:
                if len(forms) == 1:
                    forms, = forms
                result.append(forms)
            else:
                result.append(token)

        for token in tokens:
            if isinstance(token, list):
                token = token[0]
            get_in(self.lemmas_it
        return result


if __name__ == "__main__":
    from sentences import tokenized_sentences
    from pprint import pprint

    t = Translator()
    for sentence in tokenized_sentences:
        res = t.translate(sentence)
        pprint(res)
