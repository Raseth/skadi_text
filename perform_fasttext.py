import random
import time
import sys
import logging
import warnings
import numpy as np
import fastText as fasttext

from scipy.spatial.distance import cosine

from preprocess.data_preprocess import PreprocessClass
from utils.utils import bigram_path, trigram_path, npy_data_path, corpus_path, \
    wikipedia_fasttext_model, fast_text_save_path


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FastTextClass(PreprocessClass):

    def __init__(self, **kwargs):
        super(FastTextClass, self).__init__(**kwargs)
        self.category_id = kwargs.get('category_id')
        self.bigram_mod = None
        self.trigram_mod = None
        self.corpus = None
        self.fasttext_model = None
        self.id2word = None
        self.fasttext_vectors = None

    def preprocess_single_text_query(self, cleaned_text_data):
        if not self.bigram_mod or not self.trigram_mod:
            self.load_ngram_phraser(bigram_filepath=bigram_path(self.category_id),
                                    trigram_filepath=trigram_path(self.category_id))
        query_trigrams = self.make_trigrams(cleaned_text_data)
        processed_query = [self.id2word.doc2bow(text) for text in query_trigrams]
        return query_trigrams, processed_query

    def train_fasttext_skipgram(self, corpus_path,
                                output_path,
                                **kwargs):
        """
        input     training file path (required)
        output         output file path (required)
        lr             learning rate [0.05]
        lrUpdateRate change the rate of updates for the learning rate [100]
        dim            size of word vectors [100]
        ws             size of the context window [5]
        epoch          number of epochs [5]
        minCount      minimal number of word occurences [5]
        minCountLabel
        neg            number of negatives sampled [5]
        wordNgrams    max length of word ngram [1]
        loss           loss function {ns, hs, softmax} [ns]
        bucket         number of buckets [2000000]
        minn           min length of char ngram [3]
        maxn           max length of char ngram [6]
        thread         number of threads [12]
        t              sampling threshold [0.0001]
        """
        print("Training Fasttext model using Skipgram method")
        self.fasttext_model = fasttext.train_unsupervised(corpus_path, model='skipgram', **kwargs)
        self.fasttext_model.save_model(path=output_path)
        print("Model saved!")

    def load_fasttext_model(self, path):
        self.fasttext_model = fasttext.load_model(path)

    def predict_fasttext_mean_vector(self, data_words):
        """

        :param data_words: (ndarray) ['porada zakupowy', 'obuwie adidas', ...]
        :return:
        """
        fasttext_vectors = []
        for document in data_words:
            if self.fasttext_model:
                mean_vector = self.fasttext_model.get_sentence_vector(document)
                fasttext_vectors.append(mean_vector)
            else:
                sys.exit('self.fasttext_model is None!')
        return fasttext_vectors

    def predict_fasttext_multiple_data(self, doc_words):
        fasttext_vectors = self.predict_fasttext_mean_vector(doc_words)
        self.fasttext_vectors = fasttext_vectors

    def predict_random_text(self, data_words):
        tic = time.time()
        self.fasttext_vectors = self.predict_fasttext_mean_vector(data_words)
        toc = time.time()
        print(toc - tic)

        for i in range(100):
            k = random.randint(1, len(data_words))
            k = i
            # query = self.corpus[k]
            all_idxs = []

            real_text = data_words[k]
            vector = self.fasttext_vectors[k]
            distance, idx_close = self.get_closest_vectors(vector)
            print('REAL TEXT: ')
            print(vector)
            print(real_text)
            # print(original_words[k])
            print('-------------------------------------------')

            for p in range(10):
                all_idxs.append(idx_close[p])
                print('Document: ' + str(p) + ' ' + ' ; Document_id: ' + str(idx_close[p]))
                print(data_words[int(idx_close[p])])
                print(distance[p])
                # print(original_words[int(idx_close[p])])
                # print(self.fasttext_vectors[int(idx_close[p])])
                print('-------------------------------------------')

            print('All idxs: ---------------------------------')
            print(all_idxs)
            print('-------------------------------------------')

    def get_closest_vectors(self, vector):
        feats = self.fasttext_vectors
        distances = [cosine(vector, feat) for feat in feats]
        # idx_closest <- it's NOT photo_id, nor id;
        #                it's index [0, 1, 2, 3, ...] solely for sorting
        idx_closest = sorted(range(len(distances)),
                             key=lambda k: distances[k])
        distances_sorted = sorted(distances, key=float)

        return distances_sorted, idx_closest


if __name__ == "__main__":
    category_id = 1

    npzfile = np.load(npy_data_path(category_id=category_id))
    document_id = npzfile["document_id"]
    preprocessed_text = npzfile["preprocessed_text"]

    fasttext_class = FastTextClass(category_id=category_id)

    # fasttext_class.preprocess_data(preprocessed_text)

    # fasttext_class.train_fasttext_skipgram(dim=300,
    #                                        corpus_path=corpus_path(category_id=category_id),
    #                                        output_path=fast_text_save_path(category_id=category_id),
    #                                        wordNgrams=3,
    #                                        minCount=4)

    fasttext_class.load_fasttext_model(path=fast_text_save_path(category_id=category_id))
    # fasttext_class.load_fasttext_model(path=wikipedia_fasttext_model())

    fasttext_class.predict_fasttext_multiple_data(preprocessed_text)
    fasttext_class.predict_random_text(preprocessed_text)
