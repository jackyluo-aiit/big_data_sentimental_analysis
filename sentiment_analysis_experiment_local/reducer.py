#coding by Will
from operator import itemgetter
import sys
import os
import sent2vec
from collections import Counter 
#hold a dictionary
modelfile = '/home/lxq/PycharmProjects/the-disagreeable-frogs/fasttext_model/twitter_unigrams.bin'
if os.path.exists(modelfile):
    model = sent2vec.Sent2vecModel()
    model.load_model(modelfile)
    print('...model loaded!...')
else:
    print("...Model not found!...")
dic = {}
for line in sys.stdin:
  line = line.strip()
  categoriy, content = line.split('\t', 1)
  #embedding
  content = model.embed_sentence(content)
  #find current category weather existed in the dic
  if dic.get(categoriy)  == None:
    dic[categoriy] = content
  #not existed add
  else:
    dic[categoriy] += content
  #existed append

#output the collection for each category
for k,v in dic:
  print('%s\t%s' % (k, v))
