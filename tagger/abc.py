from abc import ABC, abstractmethod
from typing import List, Tuple


class PosTagger(ABC):
    @abstractmethod
    def pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        pass
