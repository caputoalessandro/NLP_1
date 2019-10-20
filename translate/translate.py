from pprint import pprint
from typing import NamedTuple, Set, List

from toolz.curried import groupby, map, mapcat, pipe

from resources import lexicon_data, lemma_translations
from utils import deepitems


class Form(NamedTuple):
    token: str
    lemma: str
    pos: str
    features: Set[str]


def unknown_form(token):
    return Form(token, token, "?", set())


MULTIWORDS = {("'", "re"): "are", ("'", "s"): "__genitive__"}


def expand_multiwords(tokens: List[str]):
    expanded = [tokens[0]]

    for token in tokens[1:]:
        multiword = MULTIWORDS.get((expanded[-1], token))
        if multiword:
            expanded.pop()
            expanded.append(multiword)
        else:
            expanded.append(token)

    return expanded


def lexicon_forms(lang: str) -> List[Form]:
    forms = []

    for lemma_id, lemma_data in lexicon_data(lang).items():
        if "forms" not in lemma_data:
            forms.append(Form(lemma_id, lemma_id, lemma_data["pos"], set()))
            continue
        for form_key, token in deepitems(lemma_data["forms"]):
            features = set(form_key)
            forms.append(Form(token, lemma_id, lemma_data["pos"], features))

    return forms


class DirectTranslator:
    def __init__(self):
        self.english_tok_to_forms = groupby(
            lambda x: x.token, lexicon_forms("en")
        )
        self.italian_lemma_to_forms = groupby(
            lambda x: x.lemma, lexicon_forms("it")
        )
        self.en_to_it = lemma_translations()

    def find_english_forms(self, token):
        return self.english_tok_to_forms.get(
            token.lower(), [unknown_form(token)]
        )

    def translate_form(self, form):
        try:
            italian_lemma = self.en_to_it[form.lemma]
        except KeyError:
            return [form]

        return [
            it_form
            for it_form in self.italian_lemma_to_forms[italian_lemma]
            if form.features.issubset(it_form.features)
        ]

    def translate_multiform(self, multiform):
        return list(mapcat(self.translate_form, multiform))

    def translate(self, tokens: List[str]):
        result = pipe(
            tokens,
            expand_multiwords,
            map(self.find_english_forms),
            list,
            map(self.translate_multiform),
            list
        )

        pprint(result)
        print()


if __name__ == "__main__":
    from sentences import tokenized_sentences

    translator = DirectTranslator()
    for sentence in tokenized_sentences:
        translator.translate(sentence)
