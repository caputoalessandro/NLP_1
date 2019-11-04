from typing import NamedTuple, Dict, List


class Form(NamedTuple):
    token: str
    lemma: str
    pos: str
    features: Dict[str, str]


Multiform = List[Form]
