import pickle
import pandas as pd
import numpy
import re
import os
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
import string
import re
import matplotlib.pyplot as plt


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
    model.save('epoch10_mincount500_window5_dm_alpha025_worker7_minalpha001_model')
    print('...saved model...')
    return model

if __name__ == '__main__':
    train_dataset = load_dataset()
    text = list(train_dataset['content'])
    test_dataset1 = train_dataset.loc[0:2, ['label','content']]
    test_dataset2 = train_dataset.loc[1599997:1599998, ['label','content']]
    test_dataset = test_dataset1.append(test_dataset2, ignore_index=True)
    print("text_dataset: \n",test_dataset.head)
    train_content = split_words(train_dataset)
    model = doc2vec_model(train_content)
    # model = Doc2Vec.load('epoch10_iter10_mincount500_window10_dm_alpha025_worker7_minalpha001_model')

    print('testing model:')
    for test_sentence in test_dataset['content'].values:
        test_text = test_sentence.split(' ')
        inferred_vector = model.infer_vector(doc_words=test_text, alpha=0.025, steps=500)
        sim_sentence = model.docvecs.most_similar([inferred_vector],topn=5)
        print("testing text:", test_text)
        for index, similarity in sim_sentence:
             # print(index, similarity)
            sentence = text[index]
            print("similar text in train_dataset: ", index, sentence, similarity)

    print('using kmeans++:')
    X = model.docvecs.vectors_docs
    kmeans_model = KMeans(n_clusters=2, init='k-means++', max_iter=100)
    kmeans_model.fit(X)
    centorids = kmeans_model.cluster_centers_
    label = kmeans_model.labels_

    pca = PCA(n_components=2).fit(X)
    datapoint = pca.transform(X)

    plt.figure
    label1 = ["r", "b"]
    color = [label1[i] for i in label]
    plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    centroidpoint = pca.transform(centorids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='k')
    plt.show()
