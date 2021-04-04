from math import log


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


def sum_values(d1, d2):
    """
    Somma i valori di un dizionario che hanno la stessa chiave.
    Ignora le chiavi che sono in un dizionario ma non nell'altro.
    """
    return {key: d1[key] + d2[key] for key in d1.keys() & d2.keys()}


def counts_to_log_likelihood(counts: dict):
    """
    Prende in input un dizionario di conteggi, e restituisce un dizionario dove come valori, al posto di ogni conteggio,
    c'è il logaritmo del conteggio diviso per il numero totale di elementi.
    In pratica, converte un dizionario di conteggi in un dizionario di log-likelihood.
    """
    denom = log(sum(counts.values()))
    return {k: log(v) - denom for k, v in counts.items()}


class DictWithMissing(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.missing = None

    def with_missing(self, missing):
        ret = DictWithMissing(self)
        ret.missing = missing
        return ret

    def __missing__(self, key):
        return self.missing
