from tokenizer import tokenize

sentences = [
    "The black droid then lowers Vader's mask and helmet onto his head."
    "These are not the droids your looking for."
    "Your friends may escape, but you are doomed."
]

tokenized_sentences = [tokenize(s) for s in sentences]
