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
