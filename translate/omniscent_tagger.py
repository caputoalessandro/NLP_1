from typing import List, Tuple

from tagger.abc import PosTagger
from resources import sentences_tags, tokenized_sentences


class OmniscentTagger(PosTagger):
    """
    Pseudo tagger che restituisce i tag annotati a mano per le frasi d'esempio.
    Utile a verificare il comportamento del traduttore quando i pos tag sono corretti.
    """

    def pos_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        for i, tokenized_sentence in enumerate(tokenized_sentences):
            if tokenized_sentence == tokens:
                return sentences_tags[i]

        raise ValueError(
            "OmniscentTagger può essere usato solo sulle frasi d'esempio."
        )
