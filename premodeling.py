import numpy as np
import tqdm

from preprocess.text_embedding_mapper import BowFasttextEmbeddingMapper
from utils.utils import text_stemming, text_normalization, npy_raw_text_data_path, npy_pos_data_path, npy_data_path


class TextProcess(BowFasttextEmbeddingMapper):

    def __init__(self, category_id, pos_tagging, fasttext_mode,
                 trigrams=True, remove_basic_stop_words=True, min_word_rarity=3, **kwargs):
        super(BowFasttextEmbeddingMapper, self).__init__(**kwargs)
        self.category_id = category_id
        self.pos_tagging = pos_tagging
        self.raw_text_data_ = None
        self.remove_basic_stop_words_ = remove_basic_stop_words
        self.min_word_rarity = min_word_rarity  # CountVect min_df for pos_tagging=False
        self.trigrams = trigrams
        self.fasttext = fasttext_mode

        if fasttext_mode:
            self.min_word_rarity = 0
            self.trigrams = False
            self.pos_tagging = False
        if pos_tagging:
            self.trigrams = False

    def data_import_(self):
        category_id = self.category_id
        RAW_DATA = npy_raw_text_data_path(category_id=category_id)

        # load data
        npzfile = np.load(RAW_DATA)

        # read data
        self.raw_text_data_ = npzfile["raw_text_data_"]

    def data_modelling_(self):
        category_id = self.category_id

        print('Preprocessing data: ')

        preprocessed_text = []
        preprocessed_pos_tags = []

        for text_row in tqdm.tqdm(self.raw_text_data_):

            normalized_text = text_normalization(data=text_row)
            stemmed_text, pos_tags = text_stemming(text=normalized_text,
                                                   remove_stop_words=self.remove_basic_stop_words_)
            preprocessed_text.append(stemmed_text)
            preprocessed_pos_tags.append(pos_tags)

        document_id = list(range(0, len(preprocessed_text)))

        if self.pos_tagging:
            self.preprocess_pos_data(data=preprocessed_text, category_id=category_id)

            full_json_data = {'document_id': document_id,
                              'preprocessed_text': preprocessed_text,
                              'preprocessed_pos_tags': preprocessed_pos_tags,
                              'raw_text_data_': self.raw_text_data_}

            np.savez(npy_pos_data_path(category_id=category_id),
                     document_id=document_id,
                     preprocessed_text=preprocessed_text,
                     preprocessed_pos_tags=preprocessed_pos_tags,
                     raw_text_data_=self.raw_text_data_)

        else:
            self.preprocess_data(data=preprocessed_text,
                                 category_id=category_id,
                                 min_words=self.min_word_rarity,
                                 trigrams=self.trigrams)

            full_json_data = {'document_id': document_id,
                              'preprocessed_text': self.fcg,
                              'raw_text_data_': self.raw_text_data_}

            np.savez(npy_data_path(category_id=category_id),
                     document_id=document_id,
                     preprocessed_text=self.fcg,
                     raw_text_data_=self.raw_text_data_)

    def text_process_(self):
        self.data_import_()
        self.data_modelling_()


if __name__ == "__main__":
    ddt = TextProcess(category_id=1,
                      pos_tagging=False,
                      fasttext_mode=False,
                      min_word_rarity=2,
                      trigrams=True)

    # ddt.text_process_()

    ddt.model_load(category=1)
    query = ddt.query_preprocess('lubię jeść kaczkę')

    a = 0
