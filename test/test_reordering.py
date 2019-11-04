from translate.pos_reordering import pos_reordering
from translate.translator import Form

"""
Il droide nero quindi abbassa la maschera 
e l'elmetto di Vader sulla sua testa

pseudoregole:
    + se il precedente è DET il successivo è NOUN
    + se il precedente è NOUN il successivo è ADJ
    + se il precedente è ADJ il successivo è ADV
    + se il precedete è ADV il successivo è VERB
    + se il precednte è X il successivo è NOUN
    + se il precedete è NOUN il successivo è CCONJ
    + se il precedente è CCONJ il successivo è NOUN elmetto
    + se il precedente è DET il successivo è DET
"""


def test_reordering():

    # fmt: off
    forms = [Form(token='il', lemma='il', pos='DET', features={'gender': 'masc', 'qty': 'singular'}),
             Form(token='black', lemma='black', pos='NOUN', features={}),
             Form(token='droide', lemma='droide', pos='NOUN', features={'gender': 'masc', 'qty': 'singular'}),
             Form(token='poi', lemma='poi', pos='ADV', features={}),
             Form(token='abbassa', lemma='abbassare', pos='VERB', features={'time': 'present', 'qty': 'singular', 'person': 2}),
             Form(token='Vader', lemma='Vader', pos='NOUN', features={}),
             Form(token='di', lemma='di', pos='PART', features={}),
             Form(token='maschera', lemma='maschera', pos='NOUN', features={'gender': 'fem', 'qty': 'singular'}),
             Form(token='e', lemma='e', pos='CCONJ', features={}),
             Form(token='elmetto', lemma='elmetto', pos='NOUN', features={'gender': 'masc', 'qty': 'singular'}),
             Form(token='onto', lemma='onto', pos='NOUN', features={}),
             Form(token='sua', lemma='suo', pos='DET', features={'gender': 'fem', 'qty': 'singular'}),
             Form(token='testa', lemma='testa', pos='NOUN', features={'gender': 'fem', 'qty': 'singular'}),
             Form(token='.', lemma='.', pos='PUNCT', features={})]
    # fmt: on

    pos_reordering(forms)
