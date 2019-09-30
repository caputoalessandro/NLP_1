import pyconll


def ud_treebank(kind: str):
    return pyconll.load_from_file(f"resources/en_partut-ud-{kind}.conllu")
