from typing import TypeVar, List, Union, Callable, Dict
from copy import copy

from toolz.curried import map, mapcat, curry

T = TypeVar("T")


def listify(obj: Union[T, List[T]]) -> List[T]:
    return obj if isinstance(obj, list) else [obj]


def subdict_matches(matchee: dict, matcher: dict):

    if matcher.keys() > matchee.keys():
        return False

    for key in matcher.keys():

        if key not in matchee:
            return False

        elif isinstance(matcher[key], dict):
            if not isinstance(matchee[key], dict):
                return False
            if not subdict_matches(matchee[key], matcher[key]):
                return False

        elif matcher[key] != matchee[key]:
            return False

    return True


@curry
def emap(fn, seq):
    """
    Versione eager di toolz.curried.map. Applica la funzione a ogni elemento della lista
    e restituisce una lista contente i risultati.
    """
    return list(map(fn, seq))


@curry
def emapcat(fn, seq):
    """
    Versione eager di toolz.curried.mapcat. Applica la funzione a ogni elemento della lista
    e restituisce una lista contenente i risultati concatenati tra loro.
    """
    return list(mapcat(fn, seq))


def _deepitems(previous_keys, obj):

    if isinstance(obj, dict):
        keys = obj.keys()
    elif isinstance(obj, list):
        keys = range(len(obj))
    else:
        yield previous_keys, obj
        return

    for key in keys:
        yield from _deepitems(previous_keys + (key,), obj[key])


def deepitems(obj):
    yield from _deepitems((), obj)


def invert(frequencies):
    result = {word: {} for words in frequencies.values() for word in words.keys()}

    for pos, words in frequencies.items():
        for word, p in words.items():
            result[word][pos] = p

    return result


def get_row(d, key):
    result = {}
    for out_k, inner_d in d.items():
        if key in inner_d:
            result[out_k] = inner_d[key]
    return result


def dict_with_missing(data, default):
    d = DictWithMissing(data)
    d.missing = default
    return d


class DictWithMissing(dict):
    def __init__(self, *args, **kwargs):
        self.missing = None
        super().__init__(*args, **kwargs)

    def __missing__(self, _):
        return self.missing
