from collections import UserDict
from typing import List, Any, Callable, Hashable


def _deepkeys(previous_keys, obj):

    if isinstance(obj, dict):
        keys = obj.keys()
    elif isinstance(obj, list):
        keys = range(len(obj))
    else:
        yield previous_keys, obj
        return

    for key in keys:
        yield from _deepkeys(previous_keys + (key,), obj[key])


def deepkeys(obj):
    yield from _deepkeys((), obj)


class DictWithMissing(dict):
    def __init__(self, *args, **kwargs):
        self.missing = None
        self.missing_factory = None
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.missing_factory:
            return self.missing_factory(key)
        return self.missing
