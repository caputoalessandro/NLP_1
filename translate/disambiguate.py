from typing import List, Set

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
                if form.features["time"] in ("infinitive", "gerund")
            ]

        result.append(next_multiform)

    return result


def disambiguate_be_when_gerund(multiforms: List[Multiform]):
    result = []

    for multiform, next_multiform in zip(multiforms, multiforms[1:]):
        if (
            all(lemma in ("essere", "stare") for lemma in multiform)
            and len(next_multiform) == 1
        ):
            lemma = (
                "stare"
                if next_multiform[0].features.get("time") == "gerund"
                else "essere"
            )
            multiform = [form for form in multiform if form.lemma == lemma]

        result.append(multiform)

    result.append(multiforms[-1])
    return result


def multiform_query(multiform: Multiform, q: dict):
    return [form for form in multiform if subdict_matches(form._asdict(), q)]


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


disambiguate = compose_left(coordinate_by_unambiguous, disambiguate_aux_verb)
