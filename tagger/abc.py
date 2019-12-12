from abc import ABC, abstractmethod
from typing import List, Tuple


class PosTagger(ABC):
    @abstractmethod
    def pos_tags(self, tokens: List[str]) -> List[str]:
        pass

    def tagged_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        return list(zip(tokens, self.pos_tags(tokens)))
