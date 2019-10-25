from typing import List
from pprint import pprint


def splitter_function(forms : List):

    initial_range = 0
    reordered_sentence = []

    for form in forms:
        if form.pos == "ADV" or form.pos == "CCONJ":
            final_range = forms.index(form)
            reordered_sentence = reordered_sentence + pos_reordering(forms[initial_range:final_range])
            initial_range = final_range

    reordered_sentence = reordered_sentence + pos_reordering(forms[initial_range:])

    pprint(reordered_sentence)
    return reordered_sentence


def pos_reordering(forms: List):

    reordered_sentence = [forms[0]]
    forms.remove(forms[0])

    while forms:

        if reordered_sentence[-1].pos == "DET" or reordered_sentence[-1].pos == "VERB":
            chosen_form = list(filter(lambda x: x.pos == "NOUN", forms))
            if chosen_form:
                reordered_sentence.append(chosen_form[0])
                forms.remove(chosen_form[0])

            else:
                reordered_sentence.append(forms[0])
                forms.remove(forms[0])

        elif reordered_sentence[-1].pos == "NOUN":
            chosen_form = list(filter(lambda x: x.pos == 'X', forms))
            if chosen_form:
                reordered_sentence.append(chosen_form[0])
                forms.remove(chosen_form[0])

            else:
                reordered_sentence.append(forms[0])
                forms.remove(forms[0])

        else:
            reordered_sentence.append(forms[0])
            forms.remove(forms[0])

    return reordered_sentence

