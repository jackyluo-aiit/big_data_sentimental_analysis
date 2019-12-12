#coding by Will
import sys
import re
# input comes from STDIN (standard input)
for line in sys.stdin:
  #splite by ,
  line = line.split(',',6)
  #get category number
  category = line[0]
  #get content
  content = line[-1]
  #remove punctuations
  content = re.sub(r'[^\w\s]','',content)
  #to lower case
  content = content.lower()
  #output
  print('%s\t%s' % (category, content))
