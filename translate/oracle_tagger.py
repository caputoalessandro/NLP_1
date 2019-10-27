from typing import List

from tagger.abc import PosTagger, TaggedToken
from sentences import tagged_sentences, tokenized_sentences


class OmniscentTagger(PosTagger):
    """
    Pseudo tagger che restituisce i tag annotati a mano per le frasi d'esempio.
    Utile a verificare il comportamento del traduttore quando i pos tag sono corretti.
    """

    def pos_tag(self, tokens: List[str]) -> List[TaggedToken]:
        for i, tokenized_sentence in enumerate(tokenized_sentences):
            if tokenized_sentence == tokens:
                return tagged_sentences[i]

        raise ValueError(
            "OmniscentTagger può essere usato solo sulle frasi d'esempio."
        )
