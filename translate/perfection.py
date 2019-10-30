from typing import List
from translate.data import Form
from itertools import  tee


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def perfection(forms: List[Form]):

    for current_form, next_form in pairwise(forms):

        if current_form.pos == "VERB" and next_form.pos == "NOUN":
            index = forms.index(next_form)
            form = Form(token='la', lemma='la', pos='X', features={'gender': 'fem', 'qty': 'singular'})
            forms.insert(index, form)

        elif current_form.pos == "CCONJ" and next_form.pos == "NOUN":
            index = forms.index(next_form)
            form = Form(token='l', lemma='l', pos='PART', features={'gender': 'masc', 'qty': 'singular'})
            forms.insert(index, form)

        elif current_form.pos == "NOUN" and next_form.pos == "PRON":
            index = forms.index(next_form)
            form = Form(token='che', lemma='che', pos='X', features={})
            forms.insert(index, form)

        elif current_form.pos == "VERB" and next_form.pos == "ADP":
            forms.remove(next_form)

    return forms


