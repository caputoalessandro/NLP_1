import pyconll
import yaml
from utils import listify
from toolz import valmap


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
