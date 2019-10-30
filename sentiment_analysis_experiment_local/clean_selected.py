import pandas as pd
import preprocessor as p
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



'''
This file is to select the data with polarity are '0' and '4';
Then clean the data, which only remains the words and punctuations.
'''
data = pd.read_csv(
    "/Users/jackyluo/OneDrive - The Chinese University of Hong Kong/Big Data/project/the-disagreeable-frogs/training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1")

data.columns = ['label', '', '', '', '', 'content']
sel_data = data[(data['label'] == 0) | (data['label'] == 4)]
print(sel_data)
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
print(new_data.head)
new_data.to_csv('sentiment_cleaned.csv')
