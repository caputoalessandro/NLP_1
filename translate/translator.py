from typing import NamedTuple, Dict, List

from toolz.curried import groupby, pipe

from resources import lexicon_data, lemma_translations
from translate.features import make_feature_dict, compatible_features
from utils import deepitems, emap, emapcat


class Form(NamedTuple):
    token: str
    lemma: str
    pos: str
    features: Dict[str, str]


Multiform = List[Form]


def unknown_form(token):
    return Form(token, token, "?", {})


MULTIWORDS = {("'", "re"): "are", ("'", "s"): "__genitive__"}


def expand_abbreviations(tokens: List[str]):
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

    for form_data in lexicon_data(lang):
        lemma = form_data["lemma"]
        pos = form_data["pos"]

        if "forms" not in form_data:
            forms.append(Form(lemma, lemma, pos, {}))
            continue
        for form_key, token in deepitems(form_data["forms"]):
            features = make_feature_dict(form_key)
            forms.append(Form(token, lemma, pos, features))

    return forms


def coordinate_by_unambiguous(multiforms: List[Multiform]):
    current_features = dict()
    result = []

    for multiform in multiforms:
        if len(multiform) == 1:
            current_features = multiform[0].features
            result.append(multiform[:])
            continue

        filtered_multiform = [
            form
            for form in multiform
            if compatible_features(
                current_features, form.features, exclude=["time"]
            )
        ]

        result.append(filtered_multiform)

    return result


class DirectTranslator:
    def __init__(self):
        self.english_tok_to_forms = groupby(
            lambda x: x.token, lexicon_forms("en")
        )
        self.italian_lemma_to_forms = groupby(
            lambda x: x.lemma, lexicon_forms("it")
        )
        self.en_to_it = lemma_translations()

    def find_multiforms_for_token(self, token):
        return self.english_tok_to_forms.get(
            token.lower(), [unknown_form(token)]
        )

    def translate_form_to_it(self, form: Form):
        try:
            italian_lemma = self.en_to_it[form.lemma]
        except KeyError:
            return [form]

        return [
            it_form
            for it_form in self.italian_lemma_to_forms[italian_lemma]
            if compatible_features(form.features, it_form.features)
        ]

    def translate_multiform_to_it(self, multiform: Multiform):
        return emapcat(self.translate_form_to_it, multiform)

    def translate(self, tokens: List[str]):
        translated_forms = pipe(
            tokens,
            expand_abbreviations,
            emap(self.find_multiforms_for_token),
            emap(self.translate_multiform_to_it),
            reversed,
            coordinate_by_unambiguous,
            reversed,
            list,
        )

        from pprint import pprint

        pprint(translated_forms)


if __name__ == "__main__":
    from sentences import tokenized_sentences

    translator = DirectTranslator()
    for sentence in tokenized_sentences:
        translator.translate(sentence)
