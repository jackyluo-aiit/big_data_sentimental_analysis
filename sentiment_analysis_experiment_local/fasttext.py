import sent2vec
import numpy as np

import re
import numpy as np
import pandas as pd
# from nltk.stem.porter import *
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import csv
import random
import fasttext
import sent2vec

if __name__ == '__main__':

    new_data = pd.read_csv("preprocessed_data.csv")
    print(new_data.loc[:, ('clean_data')])
    negative_set = new_data.loc[(new_data['label'] == 0)]
    positive_set = new_data.loc[(new_data['label'] == 4)]
    negative_setIndex = negative_set.index.values
    positive_setIndex = positive_set.index.values
    # print("negative_index:",negative_setIndex)
    # print("positive_index:",positive_setIndex)
    test_index_ne = np.random.choice(negative_setIndex, size=100)
    test_index_po = np.random.choice(positive_setIndex, size=100)
    print(test_index_ne, test_index_po)
    test_dataset_ne = new_data.loc[test_index_ne]
    test_dataset_po = new_data.loc[test_index_po]
    new_data.drop(index=test_index_ne)
    new_data.drop(index=test_index_po)
    print("after drop test set:\n", new_data)
    print("test_dataset_ne:\n", test_dataset_ne)
    print("test_dataset_po:\n", test_dataset_po)
    model = sent2vec.Sent2vecModel()
    model.load_model('/Users/jackyluo/OneDrive - The Chinese University of Hong Kong/Big '
                     'Data/project/the-disagreeable-frogs/fasttext_model/twitter_unigrams.bin')
    # emb = model.embed_sentence("once upon a time .")
    embs = []
    for sentence in new_data['clean_data'].values:
        print(sentence)
        if isinstance(sentence, str):
            emb = model.embed_sentence(sentence)
        else:
            print("not str:", sentence)
        embs.append(emb)

    embs = np.array(embs)
    embs = np.squeeze(embs)
    print(np.shape(embs))
    # print(embs)
    np.save('fast-text.npy', embs)
