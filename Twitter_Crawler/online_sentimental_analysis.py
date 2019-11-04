import os
import pandas as pd
import tweepy
import re
import string
from textblob import TextBlob
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymysql
import datetime
import time

# 'id', 'created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang',
#             'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
#             'user_mentions', 'place', 'place_coord_boundaries'


def up_load(df, company):
  sql = """INSERT INTO `crawler`(`id`, `created_at`, `source`, `original_text`, `clean_text`, `sentiment`, `polarity`, 
  `subjectivity`, `lang`, `favorite_count`, `retweet_count`, `original_author`, `possibly_sensitive`, `hashtags`, 
  `user_mentions`, `place`, `place_coord_boundaries`, `new_sentiment`, `company`) VALUES ("""

  df['original_text'].tolist()[0]
  
  for c in COLS:
    # print(df[c].tolist()[0])
    sql+= "\""+str(df[c].tolist()[0])+"\","  
    # sql+= "`"+str(df[c].tolist()[0])+"`,"
    # print(str(c)+"  : "+str(df[c]))
    if str(df[c].tolist()[0]) == "id":
      return 0

  sql +="\"-1\",\""+ str(company[1:]) +"\")"
  try:
    cursor.execute(sql)
    db.commit()
  except Exception as e:
    # print(e)
    pass
  return 0

        # print(df['id'])
#Twitter credentials for the app
consumer_key = 'FQlQCFe2KSvESyyred29PjlNj'
consumer_secret = 'CQqxq7mr9mLxAsFT6OlHNOQdPaQ2OL4KM81KM1gFNtDVn2gUib'
access_token = '1177510946064891904-b6KlxfD5BCceBLNthLO2frpkfBuebC'
access_token_secret = 'fQW42sck1OofRUGsBRasvKDNv6KINdITQ8JOzg32mJVAk'

#pass twitter credentials to tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
pd.options.display.width = 50

#file location changed to "data/telemedicine_data_extraction/" for clearer path
code = ['BTC','XNET','ATVI', 'ADBE', 'ALGN', 'ALXN', 'AMZN', 'AMGN', 'AAL', 'ADI', 'AAPL', 'AMAT', 'ASML', 'ADSK', 'ADP', 'AVGO', 'BIDU', 'BIIB', 'BMRN', 'CDNS', 'CELG', 'CERN', 'CHKP', 'CHTR', 'CTRP', 'CTAS', 'CSCO', 'CTXS', 'CMCSA', 'COST', 'CSX', 'CTSH', 'DLTR', 'EA', 'EBAY', 'EXPE', 'FAST', 'FB', 'FISV', 'GILD', 'GOOG', 'GOOGL', 'HAS', 'HSIC', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG', 'IDXX', 'JBHT', 'JD', 'KLAC', 'KHC', 'LRCX', 'LBTYA', 'LBTYK', 'LULU', 'MELI', 'MAR', 'MCHP', 'MDLZ', 'MNST', 'MSFT', 'MU', 'MXIM', 'MYL', 'NTAP', 'NFLX', 'NTES', 'NVDA', 'NXPI', 'ORLY', 'PAYX', 'PCAR', 'BKNG', 'PYPL', 'PEP', 'QCOM', 'REGN', 'ROST', 'SIRI', 'SWKS', 'SBUX', 'SYMC', 'SNPS', 'TTWO', 'TSLA', 'TXN', 'TMUS', 'ULTA', 'UAL', 'VRSN', 'VRSK', 'VRTX', 'WBA', 'WDC', 'WDAY', 'WYNN', 'XEL', 'XLNX']
# code = ['ATVI', 'ADBE']
# db = pymysql.connect("ec2-3-85-189-73.compute-1.amazonaws.com","ubuntu","123456","big_data" )
# db = pymysql.connect("3.85.189.73","ubuntu","123456","big_data" )
db = pymysql.connect("localhost","ubuntu","123456","big_data" )
cursor = db.cursor()

sql = "SELECT * FROM `crawler` WHERE `new_sentiment` = -1"
cursor.execute(sql)
results = cursor.fetchall()
a =0
for row in results:
  current_id = row[0]
  string = row[1]
  try:
    date_time_obj = datetime.datetime.strptime(string, '%a %b %d %H:%M:%S %z %Y')
  except Exception as e:
    a+=1
    continue
  date = re.sub('-', '', str(date_time_obj.date()))
  # UPDATE `crawler` SET `created_at`=2019-09-23 WHERE 'id'=1176017878610567168
  update_sql = "UPDATE `crawler` SET created_at=\""+str(date)+"\" WHERE id= "+str(current_id)
  # print(update_sql)
  cursor.execute(update_sql)
  db.commit()

  a +=1
  print(a/len(results))
  # break
