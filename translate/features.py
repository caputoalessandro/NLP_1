def classify_feature(feature):
    if feature in ("present", "infinitive", "gerundive"):
        return "time"
    elif feature in ("singular", "plural"):
        return "qty"
    elif feature in ("masc", "fem"):
        return "gender"
    elif feature in range(1, 3):
        return "person"

    raise ValueError(f"feature not classifiable: {feature}")


def make_feature_dict(features):
    return {classify_feature(feature): feature for feature in features}
