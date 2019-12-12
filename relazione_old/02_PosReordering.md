## Pos reordering

Per effettuare il reordering della frase tradotta in inglese abbiamo creato
delle regole di carattere generale basate sui POS tag. 


Quello che facciamo nella funzione `pos_reordering_local` è inserire la parola successiva in base al tag della parola
precedente.
Le regole che ci hanno permesso di riordinare le tre frasi sono due:

1. Se il POS precedente è DET o VERB, la parola successiva da inserire sarà la
   prima con POS NOUN

2. Se il POS precedente è NOUN o PRON, la parola successiva da inserire sarà la
   prima con POS PART

Se il pos precedente non rientra nei casi specificati, l'algoritmo inserirà
semplicemente la prossima parola in ordine di scorrimento.

```python 
    while forms:

        if (
            reordered_sentence[-1].pos == "DET"
            or reordered_sentence[-1].pos == "VERB"
        ):
            chosen_form = list(filter(lambda x: x.pos == "NOUN" and x.features, forms))
            if chosen_form:
                reordered_sentence.append(chosen_form[0])
                forms.remove(chosen_form[0])

            else:
                reordered_sentence.append(forms[0])
                forms.remove(forms[0])

        elif reordered_sentence[-1].pos == "NOUN" or reordered_sentence[-1].pos == "PRON":
            chosen_form = list(filter(lambda x: x.pos == "PART", forms))
            if chosen_form:
                reordered_sentence.append(chosen_form[0])
                forms.remove(chosen_form[0])

            else:
                reordered_sentence.append(forms[0])
                forms.remove(forms[0])

        else:
            reordered_sentence.append(forms[0])
            forms.remove(forms[0])

```

Per evitare che in una frase complessa l'algoritmo sposti una parola da una frase semplice a un altra
, abbiamo creato una seconda funzione `pos_reordering`. Questa funzione spezza la frase complessa
ogni volta che incontra i pos ADV e CCONJ Chiamando `pos_reordering_local` solo
sulla frase semplice, ottenendo così la frase semplice riordinata.
Le  frasi semplici riordinate quindi verranno concatenate per ottenere l'intera
frase riordinata.

```python
 for form in forms:
        if form.pos == "ADV" or form.pos == "CCONJ":
            final_range = forms.index(form)
            reordered_sentence = reordered_sentence + pos_reordering_local(
                forms[initial_range:final_range]
            )
            initial_range = final_range

    reordered_sentence = reordered_sentence + pos_reordering_local(
        forms[initial_range:]
    )

```

