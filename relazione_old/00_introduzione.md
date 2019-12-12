---
subject: "Markdown"
keywords: [Markdown, Example]
lang: "it"
...

# Traduttore direct Inglese - Italiano

## Introduzione

In  questa esercitazione costruiremo un traduttore Inglese-italiano che sfrutta
un POS tagger statistico basato su Hidden Markow Model.
In particolare implementeremo un traduttore *direct* ovvero senza
lemmatizzazione. 
Il progeetto è diviso in 4 cartelle:

1. Resources: contiene tutti i  corpus annotati che abbiamo utilizzato e due
   file yaml utilizzati per annotare i lemmi italiani e inglesi.
2. Tagger: contiene tutto cio che  riguarda ul tagger quindi la  creazione
   dell'hidden Markow Model, l'algoritmo di viterbi e un file a parte per
   gestire lo smoothing
3. Translate: contine tutte le funzioni riguardante la  traduzione come la
   disambiguazione, la creazione di features, il pos reordering, una funzione di
   perfezionamento per inserire gli articoli mancanti.
4. Test: contiene due test  rispettivamente  per il tagger e per il reordering.