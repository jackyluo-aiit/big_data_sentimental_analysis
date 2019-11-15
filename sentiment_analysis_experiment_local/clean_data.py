import pandas as pd
import preprocessor as p
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import numpy as np

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

'''
This file is to select the data with polarity are '0' and '4';
Then clean the data, which only remains the words and punctuations.
'''

# pattern = r"""(?x)                   # set flag to allow verbose regexps
# 	              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
# 	              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
# 	              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
# 	              |\.\.\.                # ellipsis
# 	              |(?:[.,;"'?():-_`])    # special characters with meanings
# 	            """

# new_data = new_data.loc[0:9, :]
# print(new_data)


# def remove_pattern(input_txt, pattern):
#     r = re.findall(pattern, input_txt)
#     for i in r:
#         input_txt = re.sub(i, '', input_txt)
#
#     return input_txt


def clean_tweet(string):
    return p.clean(string)


def clean_punc(string):
    string = string.lower()
    return re.sub("[^a-zA-Z#]+", " ", string)


def wordlemmatize(string):
    raw_words = string.split()
    wordnet_lematizer = WordNetLemmatizer()
    words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in raw_words]
    return " ".join(str(x) for x in words)


# data = pd.read_csv(
#     "training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")
# data.columns = ['label', '', '', '', '', 'content']
# sel_data = data[(data['label'] == 0) | (data['label'] == 4)]
# new_data = sel_data[['label']]
# new_data.insert(1, 'content', data['content'])

# string = 'Boeing -800 American Airlines N917NN Painted `` Air-Cal Heritage `` special colours Airliner Prof…'
# clean_line = p.clean(string)
# string = np.vectorize(remove_pattern)(string, "@[\w]*")
# new_data = new_data.loc[88019:, :]
if __name__ == '__main__':
    data = pd.read_csv("database_data.csv", encoding="utf-8")
    data.columns = ['', '', '', '', 'content', '', 'polarity', '', '', '', '', '', '', '', '', '', '', 'label', '']
    new_data = data[['label']]
    new_data.insert(1, 'content', data['content'])
    new_data.insert(2, 'polarity', data['polarity'])
    print(new_data.info)

    new_data = new_data.dropna(subset=['content'])
    new_data['clean_content'] = new_data['content'].apply(clean_tweet)
    new_data['clean_content'] = new_data['clean_content'].apply(clean_punc)
    new_data['clean_content'] = new_data['clean_content'].apply(wordlemmatize)
    new_data['clean_content'] = new_data['clean_content'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in stopwords.words('english')]))

    new_data['clean_content_length'] = new_data['clean_content'].apply(len)

    new_data.drop_duplicates(keep='first', inplace=True)
    null_index = new_data[(new_data['clean_content_length'] == 0)].index.tolist()
    new_data = new_data.drop(index=null_index)
    print(new_data.loc[:, ['clean_content', 'clean_content_length']])
    # clean_line = re.sub("[\s+\.\!\/_,$%^*(+\"\'`]+|[+——！，。？、~@#￥%……&*（）-]+", " ", clean_line)
    # clean_line = re.sub(r'\d+', ' ', clean_line)
    # print(clean_line)
    # clean_line = nltk.regexp_tokenize(clean_line, pattern)
    # print(clean_line)
    # wordnet_lematizer = WordNetLemmatizer()
    # words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in clean_line]
    # print(words)
    # filtered_words = [word for word in words if word not in stopwords.words('english')]
    # filtered_words = " ".join(str(x) for x in filtered_words)
    # print(filtered_words)

    # for index, row in new_data.iterrows():
    #     # clean_line = p.clean(line)
    #     line = row['content']
    #     line = p.clean(line)
    #     # print(line)
    #     if isinstance(line, str):
    #         clean_line = line.lower()
    #         clean_line = re.sub("[\s+\.\!\/_,$%^*()\[\]+\"\'`:;?]+|[+——！，。？、~@#￥#%……&*（）-]+", " ", clean_line)
    #         clean_line = re.sub(r'\d+', ' ', clean_line)
    #         # print(clean_line)
    #         clean_line = nltk.regexp_tokenize(clean_line, pattern)
    #         # print(clean_line)
    #         wordnet_lematizer = WordNetLemmatizer()
    #         words = [wordnet_lematizer.lemmatize(raw_word) for raw_word in clean_line]
    #         # print(words)
    #         filtered_words = [word for word in words if word not in stopwords.words('english')]
    #         filtered_words = " ".join(str(x) for x in filtered_words)
    #         print(filtered_words)
    #         new_data.loc[index, 'clean_content'] = filtered_words
    #     else:
    #         new_data.drop(index=index)
    #
    # new_data.reset_index(inplace=True, drop=True)
    # new_data.drop_duplicates(subset=['clean_content'], keep='first', inplace=True)
    # null_index = new_data[new_data.clean_content.isnull()].index.tolist()
    # new_data.drop(index=null_index)
    # new_data.dropna(axis=0, how='any', inplace=True)
    # new_data.reset_index(drop=True, inplace=True)
    #
    # print(new_data['clean_content'])
    new_data.to_csv('database_clean_data.csv')
