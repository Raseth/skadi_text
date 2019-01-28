import os
import re

from pyMorfologik import Morfologik, ListParser
from keras.preprocessing.text import text_to_word_sequence

from utils.stop_words import removed_pos, stop_words


parser = ListParser()
Morfologik = Morfologik()
punc_list = set('.;:!?/\\,#\'@$&-‘“–\xa0)(\'"')


def corpus_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/{}_corpus".format(category_id, category_id))


def rnw_corpus_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/{}_rnw_corpus".format(category_id, category_id))


def denoised_corpus_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))    
    return os.path.join(base, "../DANE/{}/{}_denoised_corpus".format(category_id, category_id))


def dictionary_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/{}_dictionary".format(category_id, category_id))


def bigram_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/{}_bigram_model".format(category_id, category_id))


def trigram_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/{}_trigram_model".format(category_id, category_id))


def count_vectorizer_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))       
    return os.path.join(base, "../DANE/{}/{}_vectorizer.pk".format(category_id, category_id))


def category_fast_text_model_path(category_id=12315):
    # fasttext_model
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/model.bin".format(category_id))


def fast_text_save_path(category_id=12315):
    # fasttext_path
    make_dirs_if_needed(category_id)    
    base = os.path.dirname(os.path.abspath(__file__))       
    return os.path.join(base, '../DANE/{}/model'.format(category_id))


def wikipedia_fasttext_model(filename="wiki.pl.bin"):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}".format(filename))


def npy_raw_text_data_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/data_{}_raw_text_data_.npz".format(category_id, category_id))


def npy_pos_data_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/data_{}_pos_txt_.npz".format(category_id, category_id))


def npy_data_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/data_{}_txt_.npz".format(category_id, category_id))


def npy_ft_vector_path(category_id=12315):
    make_dirs_if_needed(category_id)
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "../DANE/{}/{}_ft_vectors.npz".format(category_id, category_id))


def make_dirs_if_needed(category_id=12315):
    base = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(base, "../DANE/{}".format(category_id))
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def stemming_tagged(filtered_words_list):
    """
    Lematyzuje slowa i zwraca slowa razem z tagami zdaniowymi

    :param filtered_words_list: ['lubie', 'placki', 'bezsensowne_slowo']
    :return: [('lubic', verb), ('placka', subst), ('bezsensowne_slowo', )]

    adj - przymiotnik (np. „niemiecki”)
    adjp - przymiotnik poprzyimkowy (np. „niemiecku”)
    adv - przysłówek (np. „głupio”)
    conj - spójnik
    ign - ignorowana część mowy
    indecl - nieodmienna część mowy
    num - liczebnik
    pact - imiesłów przymiotnikowy czynny
    pant - imiesłów przysłówkowy uprzedni
    pcon - imiesłów przysłówkowy współczesny
    ppas - imiesłów przymiotnikowy bierny
    ppron12 - zaimek nietrzecioosobowy
    ppron3 - zaimek trzecioosobowy
    pred - predykatyw (np. „trzeba”)
    prep - przyimek
    siebie - zaimek „siebie”
    subst - rzeczownik
    verb - czasownik

    Atrybuty podstawowych form:

    sg - liczba pojedyncza
    pl - liczba mnoga
    indecl - forma nieodmienna
    irreg - forma nieregularna (nierozpoznana dokładniej pod względem wartości atrybutów, np. subst:irreg)
    nom - mianownik
    gen - dopełniacz
    acc - biernik
    dat - celownik
    inst - narzędnik
    loc - miejscownik
    voc - wołacz
    pos - stopień równy
    comp - stopień wyższy
    sup - stopień najwyższy
    m (a także, w sposób nie do końca uporządkowany, m1... m4) - rodzaj męski
    n - rodzaj nijaki
    f - rodzaj żeński
    pri - pierwsza osoba
    sec - druga osoba
    tri - trzecia osoba
    depr - forma deprecjatywna
    aff - forma niezanegowana
    neg - forma zanegowana
    refl - forma zwrotna czasownika [nie występuje w znacznikach IPI]
    perf - czasownik dokonany
    imperf - czasownik niedokonany
    ?perf - czasownik nierozpoznany pod względem aspektu
    nakc - forma nieakcentowana zaimka
    akc - forma akcentowana zaimka
    praep - forma poprzyimkowa
    npraep - forma niepoprzyimkowa
    ger - rzeczownik odsłowny
    imps - forma bezosobowa
    impt - tryb rozkazujący
    inf - bezokolicznik
    fin - forma nieprzeszła
    bedzie - forma przyszła „być”
    praet - forma przeszła czasownika (pseudoimiesłów)
    pot - tryb przypuszczający [nie występuje w znacznikach IPI]
    """
    stemmed_words = Morfologik.stem(words=filtered_words_list, parser=parser)

    ret = []
    for word in stemmed_words:
        if len(word[1]) > 0:
            forms = list(word[1].items())[0]
            ret.append((forms[0], forms[1][0].split(':')[0]))
        else:
            ret.append((word[0], ''))
    return ret


def stemming_pos_removal(filtered_words_list):
    """
    :param filtered_words_list:
    :return:
    """
    pos_words_pairs = stemming_tagged(filtered_words_list)
    ret = []
    removed_word_pairs = []
    kept_word_pairs = []
    for i in pos_words_pairs:
        if i[1] in removed_pos:
            if i not in removed_word_pairs:
                removed_word_pairs.append(i)
            continue
        else:
            ret.append((i[0], i[1]))
            if i not in kept_word_pairs:
                kept_word_pairs.append(i)
    return ret


def remove_punctuation(text):
    ret = text
    if ret is None:
        ret = ""
    translator = str.maketrans(dict.fromkeys(punc_list, " "))
    ret = ret.translate(translator)
    return ret


def text_normalization(data):
    doc = re.sub(r'\<.+?\>', ' ', data)
    doc = re.sub(r'\{.+?\}', ' ', doc)
    doc = re.sub(r'\[.+?\]', ' ', doc).lower()
    doc = remove_punctuation(doc)
    return doc


def text_stemming(text, remove_stop_words=True):
    """
    pyMorfologik - lematyzacja slow
    :return: ['pierwszy zdanie', 'drugi zdanie']
    """
    words_list = text_to_word_sequence(text)
    stemmed_words_list = stemming_pos_removal(words_list)
    if remove_stop_words:
        stem_words, stem_pos_tags = map(list, zip(*[(w[0], w[1]) for w in stemmed_words_list if not w[0] in stop_words]))
    else:
        stem_words, stem_pos_tags = map(list, zip(*[(w[0], w[1]) for w in stemmed_words_list]))

    return stem_words, stem_pos_tags


def text_preprocess(data):
    doc = text_normalization(data)
    doc = text_stemming(doc)
    return doc
