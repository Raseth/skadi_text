import sys
import pickle
import gensim
import gensim.corpora as corpora

from sklearn.feature_extraction.text import CountVectorizer

from utils.utils import text_preprocess, bigram_path, trigram_path, count_vectorizer_path, corpus_path, \
    dictionary_path, rnw_corpus_path


class PreprocessClass:

    def __init__(self, **kwargs):
        self.bigram_mod = None
        self.trigram_mod = None
        self.corpus = None
        self.id2word = None
        self.count_vectorizer = None
        self.fcg = None
        self.fast_text_model_path = kwargs.get('fast_text_model_path')
        self.count_vectorizer_path = kwargs.get('count_vectorizer_path')
        self.bigram_mod_path = kwargs.get('bigram_mod_path')
        self.trigram_mod_path = kwargs.get('trigram_mod_path')
        self.corpus_path = kwargs.get('corpus_path')
        self.rnw_corpus_path = kwargs.get('rnw_corpus_path')

    def train_ngram_phraser(self, data_words):
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
        self.bigram_mod = gensim.models.phrases.Phraser(bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(trigram)

    def save_ngram_phraser(self, bigram_filepath, trigram_filepath):
        bitemp_file = bigram_filepath
        tritemp_file = trigram_filepath
        self.bigram_mod.save(bitemp_file)
        self.trigram_mod.save(tritemp_file)

    def load_ngram_phraser(self, bigram_filepath, trigram_filepath):
        bitemp_file = bigram_filepath
        tritemp_file = trigram_filepath
        self.bigram_mod = gensim.models.phrases.Phraser.load(bitemp_file)
        self.trigram_mod = gensim.models.phrases.Phraser.load(tritemp_file)

    def make_bigrams(self, texts):
        if self.bigram_mod:
            return [self.bigram_mod[doc] for doc in texts]
        else:
            sys.exit('self.bigram_mod is None!')

    def make_trigrams(self, texts):
        if self.bigram_mod and self.trigram_mod:
            return [self.trigram_mod[self.bigram_mod[doc]] for doc in texts]
        else:
            sys.exit('self.bigram_mod or self.trigram_mod is None!')

    def query_trigrams(self, single_query):
        if self.bigram_mod and self.trigram_mod:
            return self.trigram_mod[self.bigram_mod[single_query]]
        else:
            sys.exit('self.bigram_mod or self.trigram_mod is None!')

    def train_count_vectorizer(self, **kwargs):
        """
        :param kwargs:
        min_df: int or float
        max_df: int or float
        :return:
        """
        self.count_vectorizer = CountVectorizer(**kwargs)
        document_matrix = self.count_vectorizer.fit_transform(self.corpus)
        return document_matrix

    def save_count_vectorizer(self, filepath):
        count_vectorizer_file = filepath
        with open(count_vectorizer_file, 'wb') as fin:
            pickle.dump(self.count_vectorizer, fin)

    @staticmethod
    def load_count_vectorizer(path):
        file = open(path, 'rb')
        object_file = pickle.load(file)
        file.close()
        return object_file

    def remove_noise_words(self, documents_list):
        """
        Keeps only the words used in count_vectorizer (removes too rare or too common words)
        :param documents_list: example: ['lubić jeść jeść jem', 'jeść jem', 'placki']
        :return: example: ['lubić jeść jeść', 'jeść', '']
        """
        if not self.count_vectorizer:
            sys.exit('self.count_vectorizer is None!')
        full_documents_list = [' '.join([word for word in document.split()
                                         if word in self.count_vectorizer.vocabulary_])
                               for document in documents_list]
        return full_documents_list

    @staticmethod
    def regenerate_corpora(processed_data):
        corpus = []
        print("GENERATING TEXT CORPUS")

        for processed_document in processed_data:
            full_text = ' '.join(processed_document)
            corpus.append(str(full_text))
        return corpus

    @staticmethod
    def save_corpora(text_corpora, filepath):
        import codecs
        with codecs.open(filepath, "w", "utf-8") as file:
            for item in text_corpora:
                file.write("%s\n" % '{}'.format(item))

    @staticmethod
    def load_corpora(corpus_path):
        file = open(corpus_path, 'r').read()
        corpus = file.split()
        return corpus

    def preprocess_data(self, data, category_id, min_words=3, trigrams=True):
        if trigrams:
            self.train_ngram_phraser(data)

            b_path = self.bigram_mod_path or bigram_path(category_id)
            t_path = self.trigram_mod_path or trigram_path(category_id)

            self.save_ngram_phraser(b_path, t_path)
            data = self.make_trigrams(data)

            self.corpus = self.regenerate_corpora(data)
        else:
            self.corpus = data

        c_path = self.corpus_path or corpus_path(category_id)

        self.save_corpora(text_corpora=self.corpus, filepath=c_path)

        if not min_words == 0:
            self.train_count_vectorizer(min_df=min_words)
            self.save_count_vectorizer(count_vectorizer_path(category_id=category_id))
            self.corpus = self.remove_noise_words(self.corpus)
            self.save_corpora(text_corpora=self.corpus, filepath=rnw_corpus_path(category_id=category_id))
            self.fcg = [i.split() for i in self.corpus]
        else:
            self.fcg = self.corpus

        # Create Dictionary
        self.id2word = corpora.Dictionary(self.fcg)
        self.id2word.save(dictionary_path(category_id))

        self.fcg = self.regenerate_corpora(self.fcg)

    def preprocess_pos_data(self, data, category_id):
        self.corpus = self.regenerate_corpora(data)

        c_path = self.corpus_path or corpus_path(category_id)

        self.save_corpora(text_corpora=self.corpus, filepath=c_path)

        # Create Dictionary
        self.id2word = corpora.Dictionary(data)
        self.id2word.save(dictionary_path(category_id))

    def single_query_preprocess(self, data_query, category_id, trigram=False, min_words=0):
        """
        :param data_query: [product_title, product_description, product_id]
        :return: json
        """

        # orginalny, nieobrobiony tekst
        original_doc_text = data_query[0]

        # obrobiony tekst
        query = text_preprocess(original_doc_text)
        data_words = [query.split()]

        # wczytywanie modelu do trigramow
        if not self.bigram_mod and not self.trigram_mod:
            self.load_ngram_phraser(bigram_path(category_id=category_id), trigram_path(category_id=category_id))

        # transformacja tekstu do trigramow
        data_words_trigrams_ = self.make_trigrams(data_words)

        json_data = {'preprocessed_full_document': data_words_trigrams_,
                     'original_document': original_doc_text}
        return json_data
