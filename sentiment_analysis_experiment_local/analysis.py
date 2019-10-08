# coding by Weiheng 
import pandas as pd
import numpy as np
from bert_embedding import BertEmbedding


add = "C:\\Users\\willh\\Documents\\Big Data\\Project\\sentiment140\\sentiment140.csv"
bert_embedding =  BertEmbedding()
#read csv
data = pd.read_csv(add)
data = data[0:int(len(data)*0.00001)]
print(data.shape) 


# vector = np.zeros(len(data))
vector = []
# print(len(data))
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

data['Vector'] = vector

print(data.head)
print(data.shape)

  


#merge

#signiture



# bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
#  Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
#  As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
# BERT is conceptually simple and empirically powerful. 
# It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
# sentences = bert_abstract.split('\n')
# bert_embedding = BertEmbedding()
# result = bert_embedding(sentences)

# print(result[1])
# print(result[0])