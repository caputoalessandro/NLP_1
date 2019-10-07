from typing import NamedTuple, Set, List, Union

from toolz import groupby

from resources import lexicon_data, lemma_translations
from utils import deepitems


class Form(NamedTuple):
    token: str
    lemma: str
    pos: str
    features: Set[str]


def lexicon_forms(lang: str) -> List[Form]:
    forms = []

    for lemma_id, lemma_data in lexicon_data(lang).items():
        if 'forms' not in lemma_data:
            forms.append(Form(lemma_id, lemma_id, lemma_data['pos'], set()))
            continue
        for form_key, token in deepitems(lemma_data['forms']):
            features = set(form_key)
            forms.append(Form(token, lemma_id, lemma_data['pos'], features))

    return forms


class DirectTranslator:

    def __init__(self):
        self.english_tok_to_forms = groupby(lambda x: x.token, lexicon_forms('en'))
        self.italian_lemma_to_forms = groupby(lambda x: x.lemma, lexicon_forms('it'))
        self.en_to_it = lemma_translations()

    def translate(self, tokens: List[str]):

        lexicalized_en_tokens: List[Union[Form, str]] = []

        for token in tokens:
            form_or_token = self.english_tok_to_forms.get(token.lower(), token)
            lexicalized_en_tokens.append(form_or_token)

        lexicalized_it_tokens = []

        for lex_en_token in lexicalized_en_tokens:
            if isinstance(lex_en_token, list):
                lex_it_token = []
                for form in lex_en_token:
                    try:
                        it_lemma = self.en_to_it[form.lemma]
                    except KeyError:
                        lexicalized_it_tokens.append(lex_en_token)
                        continue

                    possible_translations = [it_form for it_form in self.italian_lemma_to_forms[it_lemma] if form.features.issubset(it_form.features)]
                    lex_it_token.extend(possible_translations)

                lexicalized_it_tokens.append(lex_it_token)
            else:
                lexicalized_it_tokens.append(lex_en_token)

        from pprint import pprint
        pprint(lexicalized_it_tokens)


if __name__ == '__main__':
    from sentences import tokenized_sentences
    translator = DirectTranslator()
    for sentence in tokenized_sentences:
        translator.translate(sentence)
