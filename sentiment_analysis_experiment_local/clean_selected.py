import re
import numpy as np
import pandas as pd
import preprocessor as p
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

'''
This file is to select the data with polarity are '0' and '4';
Then clean the data, which only remains the words and punctuations.
'''
data = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1")

data.columns = ['label', '', '', '', '', 'content']
sel_data = data[(data['label'] == 0) | (data['label'] == 4)]
print("selected 0 and 4 data:", sel_data)
sel_content = sel_data['content']
content = []

for line in range(len(sel_content)):
    clean_line = p.clean(sel_content[line])
    # print(clean_line)
    content.append(clean_line)

# tfidf_vectorizer = TfidfVectorizer()
# x = tfidf_vectorizer.fit_transform(content)
# print(tfidf_vectorizer.get_feature_names())
# # print(x.toarray().shape())
#
# # print(x.shape())
# # print(data[['label', 'content']])
# vector = pd.DataFrame(x.toarray())
# print(vector)


new_data = sel_data[['label']]
new_data.insert(1, 'content', content)
print("new data:", new_data.head)



def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


new_data['clean_data'] = np.vectorize(remove_pattern)(new_data['content'], "@[\w]*")
new_data['clean_data'] = new_data['clean_data'].str.replace("[^a-zA-Z#]", "")
new_data['clean_data'] = new_data['clean_data'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
tokenized_content = new_data['clean_data'].apply(lambda x: x.split())

stemmer = PorterStemmer()
tokenized_content = tokenized_content.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range(len(tokenized_content)):
    tokenized_content[i] = ' '.join(tokenized_content[i])

new_data['clean_data'] = tokenized_content
print("after tokenized:", new_data)
print("new data len:", len(new_data))
unique_content = new_data.content.unique()
print("content len:", new_data.content.count())
print("unique content len:", len(unique_content))
unique_clean_data = new_data.clean_data.unique()
print("unique clean content len:", len(unique_clean_data))

print("Empty content:", new_data[new_data.content.isnull()])
print("Empty clean_content:", new_data[new_data.clean_data.isnull()])

new_data.drop_duplicates(subset=['clean_data'], keep='first', inplace=True)
new_data.reset_index(drop=True, inplace=True)
new_data['clean_data_length'] = new_data['clean_data'].apply(len)
print("new data:", new_data.head)









