import pyconll
import yaml


def ud_treebank(kind: str):
    return pyconll.load_from_file(f"resources/en_partut-ud-{kind}.conllu")


def lexicon_data(lang: str):
    with open(f'resources/lexicon_{lang}.yaml') as lexicon_file:
        return yaml.safe_load(lexicon_file)


def lemma_translations():
    with open('resources/en_to_it.yaml') as f:
        return yaml.safe_load(f)
