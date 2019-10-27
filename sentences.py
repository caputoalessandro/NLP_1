from tokenizer import tokenize

sentences = [
    "The black droid then lowers Vader's mask and helmet onto his head.",
    "These are not the droids you're looking for.",
    "Your friends may escape, but you are doomed.",
]

tagged_sentences = [
    [
        ("The", "DET"),
        ("black", "ADJ"),
        ("droid", "NOUN"),
        ("then", "ADV"),
        ("lowers", "VERB"),
        ("Vader", "NOUN"),
        ("'s", "PART"),
        ("mask", "NOUN"),
        ("and", "CCONJ"),
        ("helmet", "NOUN"),
        ("onto", "DET"),
        ("his", "DET"),
        ("head", "NOUN"),
        (".", "PUNCT"),
    ],
    [
        ("These", "PRON"),
        ("are", "AUX"),
        ("not", "PART"),
        ("the", "DET"),
        ("droids", "NOUN"),
        ("you", "PRON"),
        ("'re", "AUX"),
        ("looking", "VERB"),
        ("for", "ADP"),
        (".", "PUNCT"),
    ],
    [
        ("Your", "DET"),
        ("friends", "NOUN"),
        ("may", "AUX"),
        ("escape", "VERB"),
        (",", "PUNCT"),
        ("but", "CCONJ"),
        ("you", "PRON"),
        ("are", "AUX"),
        ("doomed", "VERB"),
        (".", "PUNCT"),
    ],
]

tokenized_sentences = [tokenize(s) for s in sentences]
