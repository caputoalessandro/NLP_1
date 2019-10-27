from copy import deepcopy
from typing import List

from toolz import compose_left

from translate.data import Multiform
from translate.features import compatible_features
from utils import subdict_matches

"""
Le regole di disambiguazione funzionano con l'assunzione che il POS tag in ogni Form di una
Multiform sia sempre lo stesso.
"""


def disambiguate_aux_verb(multiforms: List[Multiform]):
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


def disambiguate_be_when_gerund(multiforms: List[Multiform]):
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


def disambiguate_you_singular(multiforms: List[Multiform]):
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


def disambiguate_verb_when_pronoun(multiforms: List[Multiform]):
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


def disambiguate_last_resort(multiforms: List[Multiform]):
    return [[multiform[0]] for multiform in multiforms]


def multiform_query(multiform: Multiform, q: dict):
    return [form for form in multiform if subdict_matches(form._asdict(), q)]


def assume_third_person_if_no_pronoun(multiforms: List[Multiform]):
    result = []

    for multiform in multiforms:
        if multiform[0].pos == "VERB" and len(multiform) > 1:
            multiform = [
                form for form in multiform if form.features["person"] == 2
            ]

        result.append(multiform)

    return result


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


def assume_masc(multiforms: List[Multiform]):
    result = []
    for multiform in multiforms:
        if any(
            form.features.get("gender") == "masc" for form in multiform
        ) and any(form.features.get("gender") == "fem" for form in multiform):
            multiform = [
                form for form in multiform if form.features["gender"] == "masc"
            ]
        result.append(multiform)
    return result


def assume_third_person(multiforms: List[Multiform]):
    result = []
    for multiform in multiforms:
        if len(multiform) > 1 and multiform[0].pos == "AUX":
            multiform = [
                form
                for form in multiform
                if form.features.get("person", 2) == 2
            ]
        result.append(multiform)
    return result


def coordinate_aux_with_previous_noun(multiforms: List[Multiform]):
    result = []
    for i, multiform in enumerate(multiforms):
        if multiform[0].pos == "AUX" and len(multiform) > 1:
            prev_noun = next(
                multiform
                for multiform in multiforms[i::-1]
                if multiform[0].pos == "NOUN"
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
    disambiguate_aux_verb,
    disambiguate_be_when_gerund,
    disambiguate_you_singular,
    disambiguate_verb_when_pronoun,
    assume_masc,
    assume_third_person,
    coordinate_det_with_next_noun,
    coordinate_aux_with_previous_noun,
    disambiguate_last_resort
)
