from copy import deepcopy
from typing import List

from toolz import compose_left

from translate.data import Multiform
from translate.features import compatible_features
from utils import emap

"""
Le regole di disambiguazione funzionano con l'assunzione che il POS tag in ogni Form di una
Multiform sia sempre lo stesso.
"""


def assume_verb_time_if_previous_form_is_aux(multiforms: List[Multiform]):
    result = [multiforms[0]]

    for multiform, next_multiform in zip(multiforms, multiforms[1:]):
        if multiform[0].pos == "AUX" and next_multiform[0].pos == "VERB":
            next_multiform = [
                form
                for form in next_multiform
                if form.features["time"]
                in ("infinitive", "gerund", "pastparticiple")
            ]

        result.append(next_multiform)

    return result


def assume_be_stare_if_next_form_gerund(multiforms: List[Multiform]):
    result = []

    for multiform, next_multiform in zip(multiforms, multiforms[1:]):
        if all(form.lemma in ("essere", "stare") for form in multiform):
            lemma = (
                "stare"
                if all(
                    form.features.get("time") == "gerund"
                    for form in next_multiform
                )
                else "essere"
            )
            multiform = [form for form in multiform if form.lemma == lemma]

        result.append(multiform)

    result.append(multiforms[-1])
    return result


def assume_you_singular(multiforms: List[Multiform]):
    result = []
    for multiform in multiforms:
        if all(form.lemma == "tu" for form in multiform):
            multiform = [
                form
                for form in multiform
                if form.features["qty"] == "singular"
            ]
        result.append(multiform)
    return result


def coordinate_verb_with_pronoun(multiforms: List[Multiform]):
    result = deepcopy(multiforms)
    for i, multiform in enumerate(multiforms):
        if all(form.pos == "PRON" for form in multiform):
            pron_features = multiform[0].features.copy()
            if "gender" in pron_features:
                del pron_features["gender"]
            pron_features.setdefault("person", 2)

            try:
                i, next_verb = next(
                    (i, multiform)
                    for i, multiform in enumerate(multiforms[i:], start=i)
                    if all(form.pos in ("VERB", "AUX") for form in multiform)
                )
                result[i] = [
                    form
                    for form in next_verb
                    if compatible_features(pron_features, form.features)
                    and not form.features["time"]
                    in ("infinitive", "gerund", "pastparticiple")
                ]
            except StopIteration:
                pass
    return result


def coordinate_det_with_next_noun(multiforms: List[Multiform]):
    result = []
    for i, multiform in enumerate(multiforms):
        if multiform[0].pos == "DET":
            next_noun = next(
                multiform
                for multiform in multiforms[i:]
                if multiform[0].pos == "NOUN"
            )
            multiform = [
                form
                for form in multiform
                if any(
                    compatible_features(form.features, noun_form.features)
                    for noun_form in next_noun
                )
            ]
        result.append(multiform)
    return result


@emap
def assume_first_form(multiform: Multiform):
    return [multiform[0]]


@emap
def assume_third_person_if_no_pronoun(multiform: Multiform):
    if multiform[0].pos == "VERB" and len(multiform) > 1:
        return [form for form in multiform if form.features["person"] == 2]
    else:
        return multiform


def coordinate_by_unambiguous(multiforms: List[Multiform]):
    current_features = dict()
    result = []

    for multiform in reversed(multiforms):
        if len(multiform) == 1 and multiform[0].features:
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

    result.reverse()
    return result


@emap
def assume_masc(multiform: Multiform):
    if any(
        form.features.get("gender") == "masc" for form in multiform
    ) and any(form.features.get("gender") == "fem" for form in multiform):
        return [
            form for form in multiform if form.features["gender"] == "masc"
        ]
    else:
        return multiform


@emap
def assume_third_person(multiform: Multiform):
    if len(multiform) > 1 and multiform[0].pos == "AUX":
        return [
            form for form in multiform if form.features.get("person", 2) == 2
        ]
    else:
        return multiform


def coordinate_verb_with_subject(multiforms: List[Multiform]):
    result = []
    for i, multiform in enumerate(multiforms):
        if multiform[0].pos in ("VERB", "AUX") and len(multiform) > 1:
            prev_noun = next(
                multiform
                for multiform in multiforms[i::-1]
                if multiform[0].pos in ("NOUN", "PRON")
            )
            multiform = [
                form
                for form in multiform
                if any(
                    compatible_features(form.features, noun_form.features)
                    for noun_form in prev_noun
                )
            ]
        result.append(multiform)
    return result


disambiguate = compose_left(
    coordinate_by_unambiguous,
    assume_verb_time_if_previous_form_is_aux,
    assume_be_stare_if_next_form_gerund,
    assume_you_singular,
    coordinate_verb_with_pronoun,
    assume_masc,
    assume_third_person,
    coordinate_det_with_next_noun,
    coordinate_verb_with_subject,
    assume_first_form,
)
