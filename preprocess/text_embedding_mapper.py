from utils.utils import text_normalization, text_stemming, bigram_path, trigram_path, fast_text_save_path, count_vectorizer_path
from perform_fasttext import FastTextClass


class BowFasttextEmbeddingMapper(FastTextClass):
    def __init__(self, **kwargs):
        super(BowFasttextEmbeddingMapper, self).__init__(**kwargs)
        self.category_id = kwargs.get('category_id')
        self.fasttext = None
        self.min_word_rarity = 0
        self.trigrams = None
        self.pos_tagging = None
        self.fasttext_model = None
        self.remove_basic_stop_words_ = None

    def model_load(self, category):
        if self.fasttext:
            if self.fasttext_model is None:
                print("No fasttext model is specified. Trying to load model from path")
                ft_path = self.fast_text_model_path or fast_text_save_path(category)
                self.load_fasttext_model(path=ft_path)
        if self.min_word_rarity > 0:
            if self.count_vectorizer is None:
                cv_path = self.count_vectorizer_path or count_vectorizer_path(category)
                self.count_vectorizer = self.load_count_vectorizer(path=cv_path)
        if self.trigrams:
            if not self.bigram_mod and not self.trigram_mod:
                b_path = self.bigram_mod_path or bigram_path(category)
                t_path = self.trigram_mod_path or trigram_path(category)
                self.load_ngram_phraser(b_path, t_path)

    def query_preprocess(self, text_row):
        preprocessed_text = []
        preprocessed_pos_tags = []

        normalized_text = text_normalization(data=text_row)
        stemmed_text, pos_tags = text_stemming(text=normalized_text,
                                               remove_stop_words=self.remove_basic_stop_words_)
        preprocessed_text.append(stemmed_text)
        preprocessed_pos_tags.append(pos_tags)

        if self.pos_tagging:
            full_json_data = {'preprocessed_text': preprocessed_text,
                              'preprocessed_pos_tags': preprocessed_pos_tags}
            return full_json_data

        else:
            if self.trigrams:
                data = self.make_trigrams(preprocessed_text)
                corpus = self.regenerate_corpora(data)
            else:
                corpus = preprocessed_text

            if not self.min_word_rarity == 0:
                corpus = self.remove_noise_words(corpus)
                fcg = [i.split() for i in corpus]

            else:
                fcg = corpus

            fcg = self.regenerate_corpora(fcg)
            full_json_data = {'preprocessed_text': fcg}
            return full_json_data

    def get_query_embedding(self, text_data):
        query_preprocessed = self.query_preprocess(text_data)
        return self.predict_fasttext_mean_vector(data_words=query_preprocessed['preprocessed_text'])

    def get_batch_embedding(self, batch_data):
        return self.get_query_embedding(batch_data)
