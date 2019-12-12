#coding by Will
from operator import itemgetter
import sys
from collections import Counter 
#hold a dictionary
dic = {}
for line in sys.stdin:
  line = line.strip()
  categoriy, content = line.split('\t', 1)
  #embedding
  comtent = embedding 
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
