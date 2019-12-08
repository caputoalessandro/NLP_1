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


K = TypeVar("K")
V = TypeVar("V")


@curry
class dict_fn(Dict[K, V], Callable[[K], V]):
    def __init__(self, fn, data=None):
        super().__init__(data or {})
        self._fn = fn

    def __call__(self, key: K) -> V:
        return self[key]

    def __missing__(self, key: K) -> V:
        return self._fn(key)

    def copy(self) -> 'dict_fn[K, V]':
        return copy(self)


class DictWithMissing(dict):
    def __init__(self, *args, **kwargs):
        self.missing = None
        self.missing_factory = None
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.missing_factory:
            return self.missing_factory(key)
        return self.missing
