from typing import List

from translate.data import Form

DET_VOWEL = {
    ("masc", "singular"): "lo",
    ("masc", "plural"): "gli",
    ("fem", "singular"): "la",
    ("fem", "plural"): "le",
}

DET_CONSONANT = {
    ("masc", "singular"): "il",
    ("masc", "plural"): "i",
    ("fem", "singular"): "la",
    ("fem", "plural"): "le",
}


def make_det(form: Form):
    features = form.features
    noun_tok = form.token

    if noun_tok[0] in "aeiou" or noun_tok[0:2] == "gn":
        det = DET_VOWEL
        lemma = "il"
    else:
        det = DET_CONSONANT
        lemma = "lo"

    det_tok = det[
        features.setdefault("gender", "masc"),
        features.setdefault("qty", "singular"),
    ]

    if noun_tok[0] in "aeiou" and not features["gender"] == "plural":
        det_tok = det_tok[:-1] + "'"

    return Form(det_tok, lemma, "DET", features)


def perfection(forms: List[Form]):
    result = [forms[0]]

    for current_form, next_form in zip(forms, forms[1:]):
        needs_article = (
            current_form.pos == "VERB" and next_form.pos == "NOUN"
        ) or (current_form.pos == "CCONJ" and next_form.pos == "NOUN")

        if needs_article:
            result.append(make_det(next_form))
        elif current_form.pos == "NOUN" and next_form.pos == "PRON":
            result.append(
                Form(token="che", lemma="che", pos="CCONJ", features={})
            )

        if not (current_form.pos == "VERB" and next_form.pos == "ADP"):
            result.append(next_form)

    return result
