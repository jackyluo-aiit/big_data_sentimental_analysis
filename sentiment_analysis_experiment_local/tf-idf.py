import re
import numpy as np
import pandas as pd
import preprocessor as p
from nltk.stem.porter import *
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import csv


def distCosin(vecA, vecB):
    # # Vec = np.vstack([vecA,vecB])
    # # dist = 1 - pdist(Vec,'cosine'
    vecA = np.mat(vecA)
    vecB = np.mat(vecB)
    num = float(vecA * vecB.T)
    denom = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    if denom == 0:
        dist = 1 - num
        return dist
    cos = num / denom
    dist = 1 - cos
    # dist = np.dot(vecA, vecB) / (np.linalg.norm(vecA)*np.linalg.norm(vecB))
    # dist = 1 - spatial.distance.cosine(vecA, vecB)
    return dist


def valuate(X, po_index, ne_index, test_index):
    test_matrix = X[test_index, :]
    print("test_matrix shape:",np.shape(test_matrix))
    print(test_matrix)
    X_ne = X[ne_index, :]
    X_po = X[po_index, :]
    X_ne_mean = np.mean(X_ne, axis=0)
    X_po_mean = np.mean(X_po, axis=0)
    # print(X_ne_mean)
    label_dict = {}
    ne_list = []
    po_list = []
    for index in range(np.shape(test_matrix)[0]):
        # test_sentence = item['content']
        # test_text = test_sentence.split(' ')
        # inferred_vector = model.infer_vector(doc_words=test_text, alpha=0.025, steps=500)
        inferred_vector = test_matrix[index, :]
        # print(inferred_vector)
        # print(np.shape(inferred_vector))
        dist1 = distCosin(X_ne_mean, inferred_vector)
        dist2 = distCosin(X_po_mean, inferred_vector)
        if dist1 < dist2:
            ne_list.append(index)
        else:
            po_list.append(index)
    label_dict[0] = ne_list
    label_dict[1] = po_list
    return label_dict


def computefpp(cluster_dict, original_data, result_filename):
    original_data.reset_index(drop=True, inplace=True)
    cluster_fp_set = {}
    for key in cluster_dict.keys():
        count = 0
        cluster_set = cluster_dict[key]
        for item in cluster_set:
            if key != 0:
                key = 4
            if original_data.loc[item, ['label']].values != key:
                count += 1
            cluster_fp = count / len(cluster_set)
            cluster_fp_set[key] = cluster_fp

    print("fpp:\n", cluster_fp_set)
    with open(result_filename, 'a+') as f:
        w = csv.writer(f)
        w.writerows(cluster_fp_set.items())


if __name__ == '__main__':
    new_data = pd.read_csv("preprocessed_data.csv")
    real_data = new_data.copy()
    print(new_data.loc[:, 'clean_data'])
    negative_set = new_data.loc[(new_data['label'] == 0)]
    positive_set = new_data.loc[(new_data['label'] == 4)]
    negative_setIndex = negative_set.index.values
    positive_setIndex = positive_set.index.values
    # print("negative_index:",negative_setIndex)
    # print("positive_index:",positive_setIndex)
    test_index_ne = np.random.choice(negative_setIndex, size=100)
    test_index_po = np.random.choice(positive_setIndex, size=100)
    # print(test_index_ne, test_index_po)
    test_dataset_ne = new_data.loc[test_index_ne]
    test_dataset_po = new_data.loc[test_index_po]
    test_dataset = test_dataset_ne.append(test_dataset_po, ignore_index=False)
    test_index = test_dataset.index.values
    new_data.drop(index=test_index_ne)
    new_data.drop(index=test_index_po)
    # new_data.reset_index(drop=True)
    # negative_set = new_data.loc[(new_data['label'] == 0)]
    # positive_set = new_data.loc[(new_data['label'] == 4)]
    # negative_setIndex = negative_set.index.values
    # positive_setIndex = positive_set.index.values

    print("after drop test set:\n", new_data)
    print("test_set:", test_dataset)

    transformer = TfidfVectorizer(max_df=0.8, max_features=1000,
                                  min_df=0.001, stop_words='english',
                                  use_idf=True, tokenizer=None, ngram_range=(1, 1))
    desc_matrix = transformer.fit_transform(real_data['clean_data'].values.astype('U'))
    label_dict = valuate(desc_matrix.todense(), positive_setIndex, negative_setIndex, test_index)
    print(label_dict)
    computefpp(label_dict, test_dataset, 'mean-result_tf-idf.csv')
    # print(desc_matrix.toarray())
    # print(np.shape(desc_matrix.toarray()))