import pyconll
import yaml
from toolz import valmap

from tokenizer import tokenize
from utils import listify


def ud_treebank(kind: str):
    return pyconll.load_from_file(f"resources/en_partut-ud-{kind}.conllu")


def lexicon_data(lang: str):
    with open(f"resources/lexicon_{lang}.yaml") as lexicon_file:
        return yaml.safe_load(lexicon_file)


def lemma_translations():
    translations = {
        ("the", "DET"): ("il", "DET"),
        ("black", "ADJ"): ("nero", "ADJ"),
        ("droid", "NOUN"): ("droide", "NOUN"),
        ("then", "ADV"): ("poi", "ADV"),
        ("lower", "VERB"): ("abbassare", "VERB"),
        ("mask", "NOUN"): ("maschera", "NOUN"),
        ("and", "CCONJ"): ("e", "CCONJ"),
        ("helmet", "NOUN"): ("elmetto", "NOUN"),
        ("onto", "DET"): ("sul", "DET"),
        ("his", "DET"): ("suo", "DET"),
        ("head", "NOUN"): ("testa", "NOUN"),
        ("this", "PRON"): ("questo", "PRON"),
        ("be", "VERB"): ("essere", "VERB"),
        ("be", "AUX"): [("essere", "AUX"), ("stare", "AUX")],
        ("be_abbrev", "VERB"): ("essere", "VERB"),
        ("be_abbrev", "AUX"): [("essere", "AUX"), ("stare", "AUX")],
        ("'re", "AUX"): [("essere", "AUX"), ("stare", "AUX")],
        ("not", "PART"): ("non", "PART"),
        ("you", "PRON"): ("tu", "PRON"),
        ("look", "VERB"): ("cercare", "VERB"),
        ("for", "ADP"): ("per", "ADP"),
        ("your", "DET"): ("tuo", "DET"),
        ("friend", "NOUN"): ("amico", "NOUN"),
        ("may", "AUX"): ("potere", "AUX"),
        ("escape", "VERB"): ("fuggire", "VERB"),
        ("but", "CCONJ"): ("ma", "CCONJ"),
        ("doom", "VERB"): ("condannare", "VERB"),
        ("'s", "PART"): ("di", "PART"),
    }

    return valmap(listify, translations)


sentences = [
    "The black droid then lowers Vader's mask and helmet onto his head.",
    "These are not the droids you're looking for.",
    "Your friends may escape, but you are doomed.",
]
sentences_tags = [
    [
        "DET",
        "ADJ",
        "NOUN",
        "ADV",
        "VERB",
        "NOUN",
        "PART",
        "NOUN",
        "CCONJ",
        "NOUN",
        "DET",
        "DET",
        "NOUN",
        "PUNCT",
    ],
    ["PRON", "AUX", "PART", "DET", "NOUN", "PRON", "AUX", "VERB", "ADP", "PUNCT"],
    ["DET", "NOUN", "AUX", "VERB", "PUNCT", "CCONJ", "PRON", "AUX", "VERB", "PUNCT"],
]
tokenized_sentences = [tokenize(s) for s in sentences]
translated_sentences = [
    "Il droide nero poi abbassa la maschera e l'elmetto di Vader sulla sua testa.",
    "Questi non sono i droidi che stai cercando.",
    "I tuoi amici possono fuggire, ma tu sei condannato.",
]
