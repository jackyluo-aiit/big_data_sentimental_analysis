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

'''
This file is to select the data with polarity are '0' and '4';
Then clean the data, which only remains the words and punctuations.
'''

pattern = r"""(?x)                   # set flag to allow verbose regexps
	              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
	              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
	              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
	              |\.\.\.                # ellipsis
	              |(?:[.,;"'?():-_`])    # special characters with meanings
	            """

data = pd.read_csv("database_data.csv", encoding="utf-8")

data.columns = ['', '', '', '', 'content', '', '', '', '', '', '', '', '', '', '', '', '', 'label', '']
new_data = data[['label']]
new_data.insert(1, 'content', data['content'])
print(new_data.info)

# string = 'Boeing -800 American Airlines N917NN Painted `` Air-Cal Heritage '' special colours Airliner Prof…'
# clean_line = p.clean(string)
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

for index, row in new_data.iterrows():
    # clean_line = p.clean(line)
    line = row['content']
    # line = p.clean(line)
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
        filtered_words = " ".join(str(x) for x in filtered_words)
        print(filtered_words)
        new_data.loc[index, 'clean_content'] = filtered_words
    else:
        new_data.drop(index=index)

new_data.reset_index(inplace=True, drop=True)
new_data.drop_duplicates(subset=['clean_content'], keep='first', inplace=True)
new_data.reset_index(drop=True, inplace=True)


print(new_data['clean_content'])
new_data.to_csv('database_test.csv')
