from resources import ud_treebank


def smoothing():

    development_set = ud_treebank("dev")
    smoothing_dict = {}
    count_dict = {}

    # conto occorrenze parole
    for sentence in development_set:
        for word in sentence:
            # if not word.is_multiword():
            count_dict.setdefault(word.form, 0)
            count_dict[word.form] += 1

    # conto quante volte occorre un pos solo per le parole cche appaiono una volta
    for sentence in development_set:
        for word in sentence:
            if count_dict[word.form] == 1 and not word.is_multiword():
                smoothing_dict.setdefault(word.upos, 0)
                smoothing_dict[word.upos] += 1

    # conto quante parole singole ci sono
    single_words = sum(smoothing_dict.values())

    # calcolo le probabilità dividendo per il numero di parole singole
    for key, value in smoothing_dict.items():
        smoothing_dict[key] = value / single_words
    # print(smoothing_dict)
    return smoothing_dict
