import pickle
import pandas as pd
import numpy
import re
import os
import numpy as np
import gensim
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import string
import re
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import spatial
from numpy.linalg import norm


def load_dataset(filename='sentiment_cleaned.csv'):
    translator = str.maketrans('', '', string.punctuation)

    train_dataset = pd.read_csv('sentiment_cleaned.csv')
    train_dataset['content'] = train_dataset['content'].apply(lambda x: str(x).lower())
    train_dataset['content'] = train_dataset['content'].apply(lambda x: x.translate(translator))
    train_dataset['content'] = train_dataset['content'].apply(lambda x: re.sub('\s+', ' ', x).strip())
    print("dataset: \n", train_dataset.head)
    return train_dataset


def split_words(train_dataset):
    LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
    all_content_train = []
    j = 0
    for sentence in train_dataset['content'].values:
        all_content_train.append(LabeledSentence1(sentence, [j]))
        j += 1
    print('Num of text processed: ', j)
    return all_content_train


def doc2vec_model(train_content):
    print('...start to build doc2vec model...')
    model = Doc2Vec(epochs=10, min_count=500, window=5, dm=1, alpha=0.025, workers=7, min_alpha=0.001)
    model.build_vocab(train_content)
    print('...vocab built completed...')
    model.train(train_content, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('epoch10_mincount500_window5_dm_alpha025_worker7_minalpha001_model_v2')
    print('...saved model...')
    return model


def distCosin(vecA, vecB):
    # # Vec = np.vstack([vecA,vecB])
    # # dist = 1 - pdist(Vec,'cosine'
    vecA = np.mat(vecA)
    vecB = np.mat(vecB)
    num = float(vecA * vecB.T)
    denom = np.linalg.norm(vecA) * np.linalg.norm(vecB)
    cos = num / denom
    dist = 1 - cos
    # dist = np.dot(vecA, vecB) / (np.linalg.norm(vecA)*np.linalg.norm(vecB))
    # dist = 1 - spatial.distance.cosine(vecA, vecB)
    return dist


def get_cent(points, k):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k, n)))
    index = np.random.randint(0, m)
    cluster_centers[0, :] = np.copy(points[index, :])
    d = [0.0 for _ in range(m)]
    centroids = []
    centroids.append(index)
    for i in range(0, k - 1):
        farest = furest(cluster_centers[i, :], points, centroids)
        cluster_centers[i + 1] = np.copy(points[farest, :])
        # sum_all = 0
        # for j in range(m):
        #     d[j] = nearest(points[j, ], cluster_centers[0:i, ])
        #     sum_all += d[j]
        # sum_all *= random.random()
        # for j, di in enumerate(d):
        #     sum_all -= di
        #     if sum_all > 0:
        #         continue
        #     cluster_centers[i] = np.copy(points[j, ])
        #     break
    return cluster_centers


def get_custom_cent(points, negative_index, positive_index):
    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((2, n)))
    ne_index = np.random.choice(negative_index)
    po_index = np.random.choice(positive_index)
    cluster_centers[0, :] = np.copy(points[ne_index, :])
    cluster_centers[1, :] = np.copy(points[po_index, :])
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
    custom_centroids = createCent(dataSet, ne_index, po_index)
    clusterUpdate = True
    while clusterUpdate:
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
            pointsInClust = dataSet[np.nonzero(clusterRecord[:, 0].A == cent)[0],
                            :]  # select the nonzero rows whose index is equal to the current centroid index, and then select the corresponding rows in dataset.
            custom_centroids[cent, :] = np.mean(pointsInClust, axis=0)

    return custom_centroids, clusterRecord


if __name__ == '__main__':
    dataset = load_dataset()
    all_text = list(dataset['content'])
    test_dataset1 = dataset.loc[0:1, ['label', 'content']]
    test_dataset2 = dataset.loc[1599997:1599998, ['label', 'content']]
    test_dataset = test_dataset1.append(test_dataset2, ignore_index=False)
    train_dataset = dataset.loc[2:1599996, ['label', 'content']]
    negative_set = train_dataset.loc[(train_dataset['label'] == 0)]
    positive_set = train_dataset.loc[(train_dataset['label'] == 4)]
    negative_setIndex = negative_set.index.values
    positive_setIndex = positive_set.index.values
    print("negative_index:",negative_setIndex)
    print("positive_index:",positive_setIndex)
    print("train_dataset: \n", train_dataset.head)
    print("test_dataset: \n", test_dataset.head)
    train_content = split_words(train_dataset)
    # model = doc2vec_model(train_content)
    model = Doc2Vec.load('epoch10_mincount500_window5_dm_alpha025_worker7_minalpha001_model_v2')
    #                      'Data/project/the-disagreeable-frogs/doc2vec_model'
    #                      '/epoch10_mincount500_window5_dm_alpha025_worker7_minalpha001_model')

    print('testing model:')
    for test_sentence in test_dataset['content'].values:
        test_text = test_sentence.split(' ')
        inferred_vector = model.infer_vector(doc_words=test_text, alpha=0.025, steps=500)
        sim_sentence = model.docvecs.most_similar([inferred_vector], topn=5)
        print("testing text:", test_text)
        for index, similarity in sim_sentence:
            # print(index, similarity)
            sentence = all_text[index]
            print("similar text in train_dataset: ", index, sentence, similarity)

    print('using kmeans++:')
    X = model.docvecs.vectors_docs

    # negative = X[0]
    # negative1 = X[1]
    # positive = X[1599998]
    # # result = spatial.distance.cosine(negative, positive)
    # os_sim = distCosin(negative, positive)
    # result = distCosin(negative, negative1)
    # print("distance between %s and %s:"%(train_dataset.loc[0], train_dataset.loc[1]))
    # print(result)
    # print("distance between %s and %s:"%(train_dataset.loc[0], train_dataset.loc[1599998]))
    # print(os_sim)

    # kmeans_model = KMeans(n_clusters=2, init='k-means++', max_iter=100)
    # kmeans_model.fit(X)
    # centorids = kmeans_model.cluster_centers_
    # # closest, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, X)
    # # for index in closest:
    # #     print("The closest index of cluster: ", index)
    # #     print(train_dataset.loc[index])
    #
    # count = 0
    # cluster_fp_set = {}
    # label_dict = {i: np.where(kmeans_model.labels_ == i)[0] for i in range(kmeans_model.n_clusters)}
    # print("label_dict:", label_dict)
    # for key in label_dict.keys():
    #     count = 0
    #     cluster_set = label_dict[key]
    #     print("cluster_set:", cluster_set)
    #     for item in cluster_set:
    #         if train_dataset.loc[item, ['label']].values != key:
    #             count += 1
    #     cluster_fp = count / len(cluster_set)
    #     cluster_fp_set[key] = cluster_fp
    #
    # print("fpp: \n", cluster_fp_set)


    # label = kmeans_model.labels_
    #
    # pca = PCA(n_components=2).fit(X)
    # datapoint = pca.transform(X)
    #
    # plt.figure
    # label1 = ["r", "b"]
    # color = [label1[i] for i in label]
    # plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
    #
    # centroidpoint = pca.transform(centorids)
    # plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='k')
    # plt.show()

    # custom kmeans++
    k = 2
    custom_centroid, custom_cluster = kMeans(X, k, negative_setIndex, positive_setIndex)
    df = dfcluster = pd.DataFrame(custom_cluster, columns=['label', 'distance'])
    label_dict = {i:dfcluster[(dfcluster['label'] == i)].index for i in range(0, k)}
    print(label_dict)