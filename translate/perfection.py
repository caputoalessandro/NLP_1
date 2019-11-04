from typing import List
from translate.data import Form


def perfection(forms: List[Form]):
    result = [forms[0]]

    for current_form, next_form in zip(forms, forms[1:]):
        if current_form.pos == "VERB" and next_form.pos == "NOUN":
            result.append(
                Form(
                    token="la",
                    lemma="la",
                    pos="X",
                    features={"gender": "fem", "qty": "singular"},
                )
            )
        elif current_form.pos == "CCONJ" and next_form.pos == "NOUN":
            result.append(
                Form(
                    token="l",
                    lemma="l",
                    pos="PART",
                    features={"gender": "masc", "qty": "singular"},
                )
            )
        elif current_form.pos == "NOUN" and next_form.pos == "PRON":
            result.append(Form(token="che", lemma="che", pos="X", features={}))

        if not (current_form.pos == "VERB" and next_form.pos == "ADP"):
            result.append(next_form)

    return result
