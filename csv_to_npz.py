import tqdm
import pandas as pd
import numpy as np

from utils.utils import npy_raw_text_data_path

if __name__ == "__main__":
    RAW_DATA = './DANE/wp_posts_with_category.csv'

    category_id = 1
    OUTPUT_DATA = npy_raw_text_data_path(category_id=category_id)

    # load csv
    df = pd.read_csv(RAW_DATA, sep=",", error_bad_lines=False)
    df.fillna(value='', inplace=True)

    # transform dataframe to list
    data = df.values.tolist()
    del(RAW_DATA, df)

    raw_text_data_ = []
    document_class = []

    for row in tqdm.tqdm(data):
        raw_text_data_.append(row[1].replace("'", " "))
        document_class.append(row[2])

    # # text classes ['biznes', 'sport', 'biznes'] to int [0, 1, 0]
    # unique_classes = np.unique(document_class)
    # class_mappings = {i: j for i, j in zip(unique_classes, range(len(unique_classes)))}
    # mapped_classes = [class_mappings[k] for k in document_class]

    np.savez(OUTPUT_DATA,
             raw_text_data_=raw_text_data_,
             document_class=document_class)
a = 0
