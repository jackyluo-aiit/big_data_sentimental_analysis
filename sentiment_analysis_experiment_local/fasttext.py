import os

import sent2vec
import numpy as np

import re
import numpy as np
import pandas as pd
# from nltk.stem.porter import *
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import csv
import random
import fasttext
import sent2vec


def distCosin(vecA, vecB):
    # # Vec = np.vstack([vecA,vecB])
    # # dist = 1 - pdist(Vec,'cosine'
    vecA = np.mat(vecA)
    vecB = np.mat(vecB)
    dist = spatial.distance.cosine(vecA, vecB)
    # num = float(vecA * vecB.T)
    # denom = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    # if denom == 0:
    #     dist = 1 - num
    #     return dist
    # cos = num / denom
    # dist = 1 - cos
    # dist = np.dot(vecA, vecB) / (np.linalg.norm(vecA)*np.linalg.norm(vecB))
    # dist = 1 - spatial.distance.cosine(vecA, vecB)
    return dist


def distEclud(vecA, vecB):
  vecA = np.mat(vecA)
  vecB = np.mat(vecB)
  dist = spatial.distance.euclidean(vecA, vecB)
  return dist


def get_custom_cent(points, negative_index, positive_index):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((2, n)))
    ne_index = np.random.choice(negative_index)
    po_index = np.random.choice(positive_index)
    cluster_centers[0, :] = np.copy(points[ne_index, :])
    cluster_centers[1, :] = np.copy(points[po_index, :])
    print("Chosen centers:\n", (ne_index, po_index))
    return cluster_centers


def furest(centroid, points, centroids):
    m = np.shape(points)[0]
    max_dist = 0
    max_index = -1
    for i in range(m):
        if i in centroids:
            continue
        dist = distCosin(points[i], centroid)
        # dist = 1 - spatial.distance.cosine(points[i], centroid)
        if dist > max_dist:
            max_dist = dist
            max_index = i
    return max_index


def kMeans(dataSet, k, ne_index, po_index, distMeans=distCosin, createCent=get_custom_cent):
    m = np.shape(dataSet)[0]
    clusterRecord = np.mat(np.zeros((m, 2)))
    # centroids = createCent(dataSet, k)
    # custom_centroids = createCent(dataSet, ne_index, po_index)
    custom_centroids = createCent
    clusterUpdate = True
    iter = 1
    while clusterUpdate:
        print("iteration: ", iter)
        print("centroid: \n", np.shape(custom_centroids))
        clusterUpdate = False
        for i in range(m):  # for each point in dataset (d1,d2,dn...)==(x,y,z,...)
            # initiate distance and cluter
            minDist = np.inf
            minIndexofCluster = -1
            for clusterCent in range(k):
                dist = distMeans(dataSet[i, :], custom_centroids[clusterCent, :])
                if dist < minDist:
                    minDist = dist
                    minIndexofCluster = clusterCent

            if clusterRecord[i, 0] != minIndexofCluster:
                clusterUpdate = True
                clusterRecord[i, :] = minIndexofCluster, minDist
        # print(centroids)
        for cent in range(k):
            pointsInClust_index = np.nonzero(clusterRecord[:, 0].A == cent)[0]
            print(pointsInClust_index)
            pointsInClust = dataSet[pointsInClust_index, :]  # select the nonzero rows whose index is equal to the current centroid index, and then select the corresponding rows in dataset.
            custom_centroids[cent, :] = np.mean(pointsInClust, axis=0)
        iter += 1

    return custom_centroids, clusterRecord


def embeding(modelfile, data, matrix_file):
    if os.path.exists(matrix_file):
        return np.load(matrix_file)

    print("can not find pre-processed embedded file, start to embeding...")
    model = sent2vec.Sent2vecModel()
    model.load_model(modelfile)
    # emb = model.embed_sentence("once upon a time .")
    # data = list(data)
    embs = []
    for sentence in data:
        # print(sentence)
        if isinstance(sentence, str):
            print(sentence)
            emb = model.embed_sentence(sentence)
            # print(emb)
        else:
            print("not str:", sentence)
        embs.append(emb)
    embs = np.array(embs)
    embs = np.squeeze(embs)
    # print(np.shape(embs))
    # print(embs)
    np.save(matrix_file, embs)
    print("save to file: ", matrix_file)
    return embs


def computefpp(cluster_dict, original_data, result_filename, model=0):
    cluster_fp_set = {}
    for key in cluster_dict.keys():
        count = 0
        cluster_set = cluster_dict[key]
        if model == 0:
            for item in cluster_set:
                if key != 0:
                    key = 4
                if original_data.loc[item, ['label']].values != key:
                    count += 1
                cluster_fp = count / len(cluster_set)
                cluster_fp_set[key] = cluster_fp
        if model == 1:
            for item in cluster_set:
                if key == 1 and original_data.loc[item, ['polarity']].values <= 0:
                    count += 1
                if key == 0 and original_data.loc[item, ['polarity']].values > 0:
                    count += 1
            cluster_fp = count / len(cluster_set)

        cluster_fp_set[key] = cluster_fp
    print("fpp:\n", cluster_fp_set)
    with open(result_filename, 'a+') as f:
        w = csv.writer(f)
        # w.writerows('using re-processed dataset:')
        w.writerows(cluster_fp_set.items())


def valuate(X, po_index, ne_index, test_dataset):
    test_matrix = embeding('/Users/jackyluo/OneDrive - The Chinese University of Hong Kong/Big '
                           'Data/project/the-disagreeable-frogs/fasttext_model/twitter_unigrams.bin',
                           test_dataset['clean_content'], 'fast-text-database_test.npy')
    scaler = preprocessing.StandardScaler().fit(test_matrix)
    test_matrix_scaled = scaler.transform(test_matrix)
    print("test_matrix shape:",np.shape(test_matrix))
    X_ne = X[ne_index, :]
    X_po = X[po_index, :]
    X_ne_mean = np.mean(X_ne, axis=0)
    X_po_mean = np.mean(X_po, axis=0)
    n = np.shape(X)[1]
    X_mean = np.mat(np.zeros((2, n)))
    X_mean[0, :] = np.copy(X_ne_mean)
    X_mean[1, :] = np.copy(X_po_mean)
    # print(np.shape(X_ne))
    # print(np.shape(X_ne_mean))
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
    np.save('mean_centroid.npy', X_mean)
    return label_dict, X_mean


if __name__ == '__main__':
    new_data = pd.read_csv("sentiment_cleaned.csv")
    print("train_dataset:\n", new_data.loc[:, 'clean_content'])
    databese_test_data = pd.read_csv("database_test.csv")
    print("database_test_dataset:\n", databese_test_data.info)
    negative_set = new_data.loc[(new_data['label'] == 0)]
    positive_set = new_data.loc[(new_data['label'] == 4)]
    negative_setIndex = negative_set.index.values
    positive_setIndex = positive_set.index.values
    # print("negative_index:",negative_setIndex)
    # print("positive_index:",positive_setIndex)

    # test_index_ne = np.random.choice(negative_setIndex, size=0)
    # test_index_po = np.random.choice(positive_setIndex, size=0)
    # # print(test_index_ne, test_index_po)
    # test_dataset_ne = new_data.loc[test_index_ne]
    # test_dataset_po = new_data.loc[test_index_po]
    # test_dataset = test_dataset_ne.append(test_dataset_po, ignore_index=True)  # reset all the index in test_dataset
    # new_data.drop(index=test_index_ne)
    # new_data.drop(index=test_index_po)
    # new_data.reset_index(drop=True)
    # negative_set = new_data.loc[(new_data['label'] == 0)]
    # positive_set = new_data.loc[(new_data['label'] == 4)]
    # negative_setIndex = negative_set.index.values
    # positive_setIndex = positive_set.index.values

    print("after drop test set:\n", new_data)
    print("test_set:", databese_test_data)
    embed_matrix = embeding('/Users/jackyluo/OneDrive - The Chinese University of Hong Kong/Big '
                            'Data/project/the-disagreeable-frogs/fasttext_model/twitter_unigrams.bin',
                            new_data['clean_content'].values, 'fast-text_sentimental_train_dataset.npy')
    # scaler = preprocessing.StandardScaler().fit(embed_matrix)
    # embed_matrix_scaled = scaler.transform(embed_matrix)
    print("embeded matrix shape:", embed_matrix.shape)

    label_dict, mean_centroid = valuate(embed_matrix, positive_setIndex, negative_setIndex, databese_test_data)
    print(label_dict)
    computefpp(label_dict, databese_test_data, "mean-result_fasttext.csv", model=1)

    # print("Using sklearn kmeans++:")
    # k = 2
    # km = KMeans(n_clusters=k, init="k-means++")
    # km.fit(embed_matrix_scaled)
    # clusters = km.labels_.tolist()
    # tweets = {'Tweet': new_data["clean_data"].tolist(), 'cluster': clusters}
    # cluster_frame = pd.DataFrame(tweets)
    # cluster_frame.insert(0, column='original_text', value=new_data['content'])
    # print("after clustering:\n", cluster_frame)
    # cluster_frame.to_csv('sklearn_kmeans++_result.csv')
    # label_dict1 = {i: np.where(km.labels_ == i)[0] for i in range(km.n_clusters)}
    # computefpp(label_dict1, new_data, 'result_fpp_sklearn_kmeans++.csv')
    #
    # print("Using custom kmeans++:")
    # custom_centroid, custom_cluster = kMeans(embed_matrix, k, negative_setIndex, positive_setIndex, createCent=mean_centroid)
    # dfcluster = pd.DataFrame(custom_cluster, columns=['label', 'distance'])
    # tweets = {'Tweet': new_data["clean_content"].tolist(), 'cluster': dfcluster['label']}
    # cluster_frame = pd.DataFrame(tweets)
    # cluster_frame.insert(0, column='original_text', value=new_data['content'])
    # print("after clustering:\n", cluster_frame)
    # cluster_frame.to_csv('custom_kmeans++_result.csv')
    # label_dict2 = {i: dfcluster[(dfcluster['label'] == i)].index for i in range(0, k)}
    # computefpp(label_dict2, new_data, 'result_fpp_custom_kmeans++.csv')
