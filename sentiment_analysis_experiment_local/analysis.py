# coding by Weiheng 
import pandas as pd
import numpy as np
from bert_embedding import BertEmbedding
from collections import Counter


add = "C:\\Users\\willh\\Documents\\Big Data\\Project\\sentiment140\\sentiment140.csv"
bert_embedding =  BertEmbedding()
#read csv
data = pd.read_csv(add)

data = data[0:int(len(data)*0.00001)]
print(data.shape) 

# temp = Counter(data['tensity']) #Counter({0: 800000, 4: 248576})
# print(temp)


vector = []

#select colunms
content = data['Content']
#print(content.head())

for i in range(0, len(content)):
  sentence = content[i]
  print(sentence)
#pre-process


#word vectorlization
  result = bert_embedding(sentence)
  print(result[1][1][0].shape)
  # print(result[1][1][0])
  # vector[i] = np.array([result[1][1][0]])
  vector += [result[1][1][0]]

#merge
data['Vector'] = vector

print(data.head)
print(data.shape)
