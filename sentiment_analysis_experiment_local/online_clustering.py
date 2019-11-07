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


class online_kmeans_pipeline(object):
    def __init__(self):
        self.cluster_pd = pd.DataFrame(columns=['index', 'label'])
        self.mean_centroids = np.load('mean_centroid.npy')
        modelfile = '/Users/jackyluo/OneDrive - The Chinese University of Hong Kong/Big ' \
                    'Data/project/the-disagreeable-frogs/fasttext_model/twitter_unigrams.bin'
        if os.path.exists(modelfile):
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(modelfile)
        else:
            print("Model not found!")

    def distCosin(self, vecA, vecB):
        vecA = np.mat(vecA)
        vecB = np.mat(vecB)
        dist = spatial.distance.cosine(vecA, vecB)
        return dist

    def updataMena(self, vec, mean, n):
        new_mean = mean + (vec - mean) * (1 / n)
        return new_mean

    def preprocess_content(self, raw_content):
        preprocessed_content = ''
        pattern = r"""(?x)                   # set flag to allow verbose regexps
        	              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
        	              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
        	              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
        	              |\.\.\.                # ellipsis
        	              |(?:[.,;"'?():-_`])    # special characters with meanings
        	            """

        line = raw_content
        line = p.clean(line)
        # print(line)
        if isinstance(line, str):
            clean_line = line.lower()
            clean_line = re.sub("[\s+\.\!\/_,$%^*(+\"\'`:;?]+|[+——！，。？、~@#￥#%……&*（）-]+", " ", clean_line)
            clean_line = re.sub(r'\d+', ' ', clean_line)
            # print(clean_line)
            clean_line = nltk.regexp_tokenize(clean_line, pattern)
            # print(clean_line)
            wordnet_lematizer = WordNetLemmatizer()
            words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in clean_line]
            # print(words)
            filtered_words = [word for word in words if word not in stopwords.words('english')]
            preprocessed_content = " ".join(str(x) for x in filtered_words)
            preprocessed_content = preprocessed_content
            print("after preprocessed:", preprocessed_content)
            return preprocessed_content
        else:
            return -1

    def embed_content(self, preprocessed_content):
        vectorized_content = np.zeros((1, 700))
        if isinstance(preprocessed_content, str):
            # modelfile = '/Users/jackyluo/OneDrive - The Chinese University of Hong Kong/Big ' \
            #             'Data/project/the-disagreeable-frogs/fasttext_model/twitter_unigrams.bin'
            # if os.path.exists(modelfile):
            #     model = sent2vec.Sent2vecModel()
            #     model.load_model(modelfile)
            # else:
            #     return print('can not find any model!')
            print("start to embeding...")
            emb = self.model.embed_sentence(preprocessed_content)
            # print(emb)
            vectorized_content = np.array(emb)
            print(np.shape(vectorized_content))
            with open("vectorized.npy", 'a+') as f:
                # f = file("vectorized.npy", 'a')
                np.save("vectorized.npy", vectorized_content)
                f.close()
                print("save to file: ", 'vectorized.npy')
            return vectorized_content
        else:
            print('It is not a string!')
            return -1

    def kmeanspp(self, vectorized_content, in_index):
        input_vec = vectorized_content
        ne_mean = np.copy(self.mean_centroids[0, :])
        po_mean = np.copy(self.mean_centroids[1, :])
        ne_mean = np.array(ne_mean)
        po_mean = np.array(po_mean)
        print("ne_mean:", np.shape(ne_mean))
        print("po_mean:", np.shape(po_mean))
        print("vectorized_content:", np.shape(input_vec))
        dist_ne = self.distCosin(ne_mean, input_vec)
        dist_po = self.distCosin(po_mean, input_vec)
        self.cluster_pd.loc[in_index, 'index'] = in_index
        if dist_ne <= dist_po:
            self.cluster_pd.loc[in_index, 'label'] = 0
            n = self.cluster_pd.loc[self.cluster_pd['label'] == 0].shape[0]
            print(n)
            self.mean_centroids[0, :] = self.updataMena(vectorized_content, ne_mean, n)
        else:
            self.cluster_pd.loc[in_index, 'label'] = 1
            n = self.cluster_pd.loc[self.cluster_pd['label'] == 1].shape[0]
            print(n)
            self.mean_centroids[1, :] = self.updataMena(vectorized_content, ne_mean, n)

    def process_content(self, raw_content):
        df = raw_content
        for index, row in df.iterrows():
            preprocessed_content = self.preprocess_content(row['content'])
            vectorized_content = self.embed_content(preprocessed_content)
            in_index = row['in_index']
            self.kmeanspp(vectorized_content, in_index)
        # self.cluster_pd.to_csv('cluster_result.csv', mode='a+', header=False)
        return self.cluster_pd


    def saveresult(self):
        self.cluster_pd.to_csv('cluster_result.csv', mode='a+', header=False)
        print("save to file: cluster_result.csv")
