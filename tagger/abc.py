from abc import ABC, abstractmethod


class PosTagger(ABC):
    @abstractmethod
    def pos_tags(self, tokens: list[str]) -> list[str]:
        pass

    def tagged_tokens(self, tokens: list[str]) -> list[tuple[str, str]]:
        return list(zip(tokens, self.pos_tags(tokens)))
