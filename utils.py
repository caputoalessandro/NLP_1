
def transpose(outer):
    result = {k2: {} for inner in outer.values() for k2 in inner.keys()}

    for k1, inner in outer.items():
        for k2, val in inner.items():
            result[k2][k1] = val

    return result


def get_row(outer, key):
    result = {}
    for k1, inner in outer.items():
        if key in inner:
            result[k1] = inner[key]
    return result


def dict_with_missing(data, default):
    d = DictWithMissing(data)
    d.missing = default
    return d


def merge_with(fn, d1, d2):
    """
    Applica fn ai valori che hanno la stessa chiave in d1 e d2.
    Restituisce un dict dove per ogni chiave in comune il valore è il risultato della funzione.

    >>> from operator import add
    >>> merge_with(add, {'a': 1, 'b': 2, 'c': 2}, {'a': 2, 'b': 2})
    {'a': 3, 'b': 4}
    """
    return {key: fn(d1[key], d2[key]) for key in d1.keys() & d2.keys()}


class DictWithMissing(dict):
    def __init__(self, *args, **kwargs):
        self.missing = None
        super().__init__(*args, **kwargs)

    def __missing__(self, _):
        return self.missing
