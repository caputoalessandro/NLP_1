from translate.translator import Form
from translate.pos_reordering import splitter_function

def test_reordering():

    forms = [Form(token='il', lemma='il', pos='DET', features={}),
             Form(token='nero', lemma='nero', pos='ADJ', features={}),
             Form(token='droide', lemma='droide', pos='NOUN', features={}),
             Form(token='poi', lemma='poi', pos='ADV', features={}),
             Form(token='abbassa', lemma='abbassare', pos='VERB', features={}),
             Form(token='Vader', lemma='Vader', pos='PROPN', features={}),
             Form(token='di', lemma='di', pos='X', features={}),
             Form(token='maschera', lemma='maschera', pos='NOUN', features={}),
             Form(token='se', lemma='e', pos='CCONJ', features={}),
             Form(token='elmetto', lemma='elmetto', pos='NOUN', features={}),
             Form(token='sulla', lemma='sul', pos='DET', features={}),
             Form(token='sua', lemma='suo', pos='DET', features={}),
             Form(token='testa', lemma='testa', pos='NOUN', features={}),
             Form(token='.', lemma='.', pos='?', features={})]

    splitter_function(forms)

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