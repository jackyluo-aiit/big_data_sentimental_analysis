#coding by Will

from operator import itemgetter
import sys
from collections import Counter 
#hold a dictionary
dic = {}
for line in sys.stdin:
  line = line.strip()
  word, count = line.split('\t', 1)
  #find current word in dic
  if dic.get(word)  == None:
    dic[word] = 0
  #not existed add
  else:
    dic[word] += 1
  #existed ++1

most_common = Counter(dic).most_common(200)

#pop the 200 largest ones according to values
for m in most_common:
  print('%s\t%s' % (m[0], m[1]))
