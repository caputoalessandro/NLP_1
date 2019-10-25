from typing import Dict, Set

from toolz import pluck


def classify_feature(feature):
    if feature in ("present", "infinitive", "gerund", "pastparticiple"):
        return "time"
    elif feature in ("singular", "plural"):
        return "qty"
    elif feature in ("masc", "fem"):
        return "gender"
    elif feature in range(3):
        return "person"

    raise ValueError(f"feature not classifiable: {feature}")


def make_feature_dict(features):
    return {classify_feature(feature): feature for feature in features}


def compatible_features(f1: Dict[str, str], f2: Dict[str, str], exclude=()):
    keys_to_check = set(f1.keys() & f2.keys())
    keys_to_check.difference_update(exclude)

    f1_intersection, f2_intersection = pluck(list(keys_to_check), (f1, f2))
    return f1_intersection == f2_intersection
