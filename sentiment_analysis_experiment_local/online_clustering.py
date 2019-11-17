import pandas as pd
import preprocessor as p
import re
import string
import numpy as np
import nltk
import os
from scipy import spatial

nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import sent2vec
from sentiment_analysis_experiment_local.clean_data import *


class online_kmeans_pipeline(object):
    def __init__(self):
        self.cluster_pd = pd.DataFrame(columns=['label'])
        self.mean_centroids = np.load('mean_centroid.npy')
        modelfile = 'twitter_unigrams.bin'
        if os.path.exists(modelfile):
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(modelfile)
            print('...model loaded!...')
        else:
            print("...Model not found!...")

    def distCosin(self, vecA, vecB):
        vecA = np.mat(vecA)
        vecB = np.mat(vecB)
        dist = spatial.distance.cosine(vecA, vecB)
        return dist

    def updateMean(self, vec, mean, n):
        new_mean = mean + (vec - mean) * (1 / n)
        return new_mean

    def preprocess_content(self, raw_content):
        line = raw_content
        # print(line)
        if isinstance(line, str):
            line = p.clean(line)
            clean_line = clean_tweet(line)
            clean_line = clean_punc(clean_line)
            clean_line = wordlemmatize(clean_line)
            # clean_line = line.lower()
            # clean_line = re.sub("[\s+\.\!\/_,$%^*(+\"\'`:;?]+|[+——！，。？、~@#￥#%……&*（）-]+", " ", clean_line)
            # clean_line = re.sub(r'\d+', ' ', clean_line)
            # # print(clean_line)
            # clean_line = nltk.regexp_tokenize(clean_line, pattern)
            # # print(clean_line)
            # wordnet_lematizer = WordNetLemmatizer()
            # words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in clean_line]
            # print(words)
            words = clean_line.split()
            filtered_words = [word for word in words if word not in stopwords.words('english')]
            preprocessed_content = " ".join(str(x) for x in filtered_words)
            preprocessed_content = preprocessed_content
            return preprocessed_content
        else:
            print(line, ': It is not a string!')
            return -1

    def embed_content(self, preprocessed_content):
        vectorized_content = np.zeros((1, 700))
        if isinstance(preprocessed_content, str):
            print("start to embeding...")
            emb = self.model.embed_sentence(preprocessed_content)
            # print(emb)
            vectorized_content = np.array(emb)
            #             print('embedded vector:', np.shape(vectorized_content))
            #             with open("vectorized.npy", 'a+') as f:
            #                 # f = file("vectorized.npy", 'a')
            #                 np.save("vectorized.npy", vectorized_content)
            #                 f.close()
            #                 print("save to file: ", 'vectorized.npy')
            return vectorized_content
        else:
            return -1

    def kmeanspp(self, vectorized_content, in_index):
        input_vec = vectorized_content
        ne_mean = np.copy(self.mean_centroids[0, :])
        po_mean = np.copy(self.mean_centroids[1, :])
        ne_mean = np.array(ne_mean)
        po_mean = np.array(po_mean)
        #         print("ne_mean:", np.shape(ne_mean))
        #         print("po_mean:", np.shape(po_mean))
        print("vectorized_content:", np.shape(input_vec))
        dist_ne = self.distCosin(ne_mean, input_vec)
        dist_po = self.distCosin(po_mean, input_vec)
        tmp_cluster = pd.DataFrame(columns=['label'])
        if dist_ne <= dist_po:
            self.cluster_pd.loc[in_index, 'label'] = 0
            tmp_cluster.loc[in_index, 'label'] = 0
            n = self.cluster_pd.loc[self.cluster_pd['label'] == 0].shape[0]
            print('number of elements in negative cluster:', n)
            self.mean_centroids[0, :] = self.updateMean(vectorized_content, ne_mean, n)
        else:
            self.cluster_pd.loc[in_index, 'label'] = 1
            tmp_cluster.loc[in_index, 'label'] = 1
            n = self.cluster_pd.loc[self.cluster_pd['label'] == 1].shape[0]
            print('number of elements in positive cluster:', n)
            self.mean_centroids[1, :] = self.updateMean(vectorized_content, po_mean, n)
        tmp_cluster.to_csv('tmp_result.csv', mode='a+', header=False)
        return tmp_cluster

    def process_content(self, raw_content):
        df = raw_content
        for index, row in df.iterrows():
            preprocessed_content = self.preprocess_content(row['content'])
            if isinstance(preprocessed_content, str):
                print("after preprocessed:", preprocessed_content)
                if len(preprocessed_content.split()) >= 3:
                    vectorized_content = self.embed_content(preprocessed_content)
                    in_index = row['in_index']
                    return self.kmeanspp(vectorized_content, in_index)
                else:
                    print('Less than 3 words left after preprocessed!!!')

    def saveresult(self):
        self.cluster_pd.to_csv('cluster_result.csv', mode='a+', header=False)
        print("save to file: cluster_result.csv")
