from abc import ABC, abstractmethod
from typing import List


class PosTagger(ABC):
    @abstractmethod
    def pos_tag(self, tokens: List[str]):
        pass
