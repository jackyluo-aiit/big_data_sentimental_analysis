import pandas as pd
import preprocessor as p
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
import numpy as np

nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

'''
This file is to select the data with polarity are '0' and '4';
Then clean the data, which only remains the words and punctuations.
'''


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


pattern = r"""(?x)                   # set flag to allow verbose regexps
	              (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
	              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
	              |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
	              |\.\.\.                # ellipsis
	              |(?:[.,;"'?():-_`])    # special characters with meanings
	            """

data = pd.read_csv(
    "/Users/jackyluo/OneDrive - The Chinese University of Hong Kong/Big Data/project/the-disagreeable-frogs/training.1600000.processed.noemoticon.csv",
    encoding="ISO-8859-1")

data.columns = ['label', '', '', '', '', 'content']
sel_data = data[(data['label'] == 0) | (data['label'] == 4)]
new_data = sel_data[['label']]
new_data.insert(1, 'content', data['content'])

# new_data.loc[:, ['clean_content']] = np.vectorize(remove_pattern)(new_data.loc[:, ['content']], "@[\w]*")
# new_data.loc[:, 'clean_content'] = new_data.loc[:, 'content'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
content = []

# string = 'Happy 38th Birthday to my boo of alll time!!! Tupac Amaru Shakur'
# clean_line = p.clean(string)
# clean_line = nltk.regexp_tokenize(clean_line, pattern)
# # raw_words = nltk.word_tokenize(clean_line)
# filtered_words = [word for word in clean_line if word not in stopwords.words('english')]
# filtered_words = " ".join(str(x) for x in filtered_words)
# print(filtered_words)
for line in new_data['content'].values:
    clean_line = p.clean(line)
    clean_line = nltk.regexp_tokenize(clean_line, pattern)
    # raw_words = nltk.word_tokenize(clean_line)
    filtered_words = [word for word in clean_line if word not in stopwords.words('english')]
    filtered_words = " ".join(str(x) for x in filtered_words)
    content.append(filtered_words)

new_data.loc[:, 'clean_content'] = content
new_data['data_length'] = new_data['clean_data'].apply(len)
new_data.loc[:, 'clean_content'] = new_data[(new_data['data_length'] >= 3)]

print(new_data.clean_content)
new_data.to_csv('sentiment_cleaned.csv')
