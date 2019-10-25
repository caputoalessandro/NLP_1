from abc import ABC, abstractmethod
from typing import List, Tuple


TaggedToken = Tuple[str, str]


class PosTagger(ABC):
    @abstractmethod
    def pos_tag(self, tokens: List[str]) -> List[TaggedToken]:
        pass
