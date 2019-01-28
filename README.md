# skadi_text
A set of NLP tools for machine learning projects, specifically in Polish language.


# Installation
Requires Python >3.6 with preinstalled pipenv (it is best to use Anaconda, since it has preinstalled BLAS/Lapack, 
which may be required for sklearn).

For pipenv installation, check (https://pypi.org/project/pipenv/).

Navigate to main folder, and use:
```buildoutcfg
pipenv shell
pipenv install
```

All required dependencies, should get installed, however there is possibility that fasttext library gets outdated.
The version I use is (https://github.com/facebookresearch/fastText) from January 2019.

If you're NOT using Anaconda, but a clean new Python installation, there is a possibility, 
you won't have BLAS/Lapack installed, which will result in failure, when trying to import sklearn.
If this happens, try:
```
sudo apt-get install gfortran
sudo apt-get update

sudo apt-get install libblas-dev checkinstall
sudo apt-get install libblas-doc checkinstall
sudo apt-get install liblapacke-dev checkinstall
sudo apt-get install liblapack-doc checkinstall
```

Full details, at (https://askubuntu.com/questions/623578/installing-blas-and-lapack-packages).

# Usage

Turn on prepared enviroment, via:
```
pipenv shell
```

If you want to use pretrained fasttext model, download one from (https://fasttext.cc/docs/en/crawl-vectors.html).
In default, the model should be renamed to "wiki.pl.bin" and placed inside "DANE" folder. 

The input data should be in .npz format. If you have it in .csv format file, check csv_to_npz.py file, for a tutorial.

After that, use premodelling.py file, for a tutorial, how to transform data.

There are 3 modes (first for semantic modelling preprocess):

SEMANTIC PREPROCESSING:
- category_id - id of a folder in ./DANE/
- pos_tagging - True - Use True, if you want to stem text data and get it's POS tags.
- remove_basic_stop_words - (True/False) - remove basic stop words, specified inside utils/stop_words.py, 
however it may be better to keep these words
- fasttext_mode - False

RETRIEVING WORD EMBEDDINGS:
- category_id - id of a folder in ./DANE/
- pos_tagging - False 
- remove_basic_stop_words - (True/False)
- fasttext_mode - True

It will preprocess data, for fasttext to be used. After that, use train_fasttext_skipgram method, 
with specified set of parameters.

CUSTOM WORD EMBEDDINGS PREPROCESS:
- category_id - id of a folder in ./DANE/
- pos_tagging - False 
- remove_basic_stop_words - (True/False)
- fasttext_mode - False
- trigrams - (True/False)
- min_word_rarity - 0/1/2/...

Additionally, it is possible to further remove stop words from text, in any of these modes, by its POS tag. 
To do so, add specific tags, you wish to be removed in /utils/stop_words.py in 'removed_pos'.


