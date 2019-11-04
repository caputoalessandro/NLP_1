from typing import List

from toolz.curried import groupby, pipe, concat, reduce

from resources import lexicon_data, lemma_translations, tokenized_sentences
from tagger import ud_viterbi_tagger, PosTagger
from translate.data import Form, Multiform
from translate.disambiguate import disambiguate
from translate.features import make_feature_dict, compatible_features
from translate.omniscent_tagger import OmniscentTagger
from translate.perfection import perfection
from translate.pos_reordering import pos_reordering
from utils import deepitems, emap, emapcat


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


def join_form_to_sentence(sentence: str, form):
    no_space = len(sentence) == 0 or sentence[-1] == "'" or form.pos == "PUNCT"
    sep = "" if no_space else " "
    return sep.join((sentence, form.token))


def forms_to_sentence(forms: List[Form]) -> str:
    return reduce(join_form_to_sentence, forms, "")


class DirectTranslator:
    def __init__(self, tagger: PosTagger):
        self.english_tok_to_forms = groupby(
            lambda x: (x.token, x.pos), lexicon_forms("en")
        )
        self.italian_lemma_to_forms = groupby(
            lambda x: (x.lemma, x.pos), lexicon_forms("it")
        )
        self.en_to_it = lemma_translations()
        self.tagger = tagger

    def find_multiforms_for_token(self, tagged_token):
        token, pos = tagged_token
        return self.english_tok_to_forms.get(
            (token.lower(), pos), [Form(token, token, pos, {})]
        )

    def translate_form_to_it(self, form: Form):
        try:
            italian_lemmas = self.en_to_it[form.lemma, form.pos]
        except KeyError:
            return [form]

        return [
            it_form
            for italian_lemma in italian_lemmas
            for it_form in self.italian_lemma_to_forms[italian_lemma]
            if compatible_features(form.features, it_form.features)
        ]

    def translate_multiform_to_it(self, multiform: Multiform):
        return emapcat(self.translate_form_to_it, multiform)

    def translate(self, tokens: List[str]):
        return pipe(
            tokens,
            self.tagger.pos_tag,
            emap(self.find_multiforms_for_token),
            emap(self.translate_multiform_to_it),
            disambiguate,
            concat,
            list,
            pos_reordering,
            perfection,
            forms_to_sentence,
            lambda sentence: sentence[0].upper() + sentence[1:],
        )


def main():
    viterbi_tagger = ud_viterbi_tagger()
    viterbi_translator = DirectTranslator(viterbi_tagger)
    omniscent_translator = DirectTranslator(OmniscentTagger())

    for tokens in tokenized_sentences:
        print("--- VITERBI ---")
        print(viterbi_translator.translate(tokens))
        print("--- OMNISCENT ---")
        print(omniscent_translator.translate(tokens))


if __name__ == "__main__":
    main()
