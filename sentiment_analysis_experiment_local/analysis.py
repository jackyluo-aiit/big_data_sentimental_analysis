# coding by Weiheng 
import pandas as pd
import numpy as np
from bert_embedding import BertEmbedding
from collections import Counter
import preprocessor as p
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


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
  print("    ")
  sentence = content[i]
  print(sentence)
  print(len(sentence))
  sentence = p.clean(sentence).split('\n')
  print("    ")
  print(sentence)

#pre-process

#word vectorlization/ tokenlize
  result = bert_embedding(sentence)
  print(result)
  print("  ")
  # print(len(result[0][1]))
  # print(np.shape(np.array(result[0][1])))
  vector += [np.array(result[0][1])]
  # break
#merge
data['Vector'] = vector

print(data.head)
print(data.shape)
