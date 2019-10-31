# Dylan's MacbookPro #
# JP DILAN KALPA - 11634268 #
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

#mrhod clean_tweets()
def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)

    #after tweepy preprocessing the colon left remain after removing mentions
    #or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)


    #remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
    #print(word_tokens)
    #print(filtered_sentence)


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

#method write_tweets()
def write_tweets(keyword, file):
    # If the file exists, then read the existing data from the CSV file.
    print("holding keyword "+str(keyword))
    print("cralwing from "+str(start_date)+"  to  "+str(end_date))
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.DataFrame(columns=COLS)
    #page attribute in tweepy.cursor and iteration
    a = 0
    for page in tweepy.Cursor(api.search, q=keyword,count=200, include_rts=False, since=start_date).pages(50):
        a += 1
        print("cralwing page "+str(a))
        for status in page:
            new_entry = []
            status = status._json
            if status['lang'] != 'en':## check whether the tweet is in english or skip to the next tweet
                continue
            #when run the code, below code replaces the retweet amount and
            #no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                   status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue
            clean_text = p.clean(status['text'])    #tweepy preprocessing called for basic preprocessing
            filtered_tweet=clean_tweets(clean_text) #call clean_tweet method for extra preprocessing
            blob = TextBlob(filtered_tweet) #pass textBlob method for sentiment calculations
            Sentiment = blob.sentiment
            polarity = Sentiment.polarity  #seperate polarity and subjectivity in to two variables
            subjectivity = Sentiment.subjectivity

            #new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['text'],filtered_tweet, Sentiment,polarity,subjectivity, status['lang'],
                          status['favorite_count'], status['retweet_count']]

            #to append original author of the tweet
            new_entry.append(status['user']['screen_name'])

            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)

            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)

            #get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)
            try:
                coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            new_entry.append(coordinates)

            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            # print(single_tweet_df['id'])
            df = df.append(single_tweet_df, ignore_index=True)

            # csvFile = open(file, 'a' ,encoding='utf-8')
            # df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")
        
        df['source'] = 'nan'
        df['original_text'] = 'drop'
        for i in range(0,len(df)):
          up_load(df.iloc[[i],:],keyword)
          # print(df.iloc[[i],:])
          # print("*********")
        # print(df.head())
    #end of page
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
db = pymysql.connect("localhost","ubuntu","123456","big_data" )
cursor = db.cursor()


print("*************")
date = datetime.date.today() - datetime.timedelta(1)
print(str(date))
print("*************")

days_from_now = 50
#set two date variables for date range
# start_date = '2018-10-01'
# end_date = '2019-09-28'

while True:

    end_date = datetime.date.today() - datetime.timedelta(days_from_now)
    start_date = datetime.date.today() - datetime.timedelta(days_from_now+1)
    days_from_now -= 1

    for i in code:
        telemedicine_tweets = i +"_data.csv"

        #columns of the csv file
        COLS = ['id', 'created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang',
                'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
                'user_mentions', 'place', 'place_coord_boundaries']


        # Happy Emoticons
        emoticons_happy = set([
            ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
            ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
            '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
            'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
            '<3'
            ])

        # Sad Emoticons
        emoticons_sad = set([
            ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
            ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
            ':c', ':{', '>:\\', ';('
            ])

        #Emoji patterns
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

        #combine sad and happy emoticons
        emoticons = emoticons_happy.union(emoticons_sad)

        #declare keywords as a query for three categories
        # telemedicine_keywords = '#iPhone11 OR #iphone11'
        telemedicine_keywords = '#%s'%i

        write_tweets(telemedicine_keywords,  telemedicine_tweets)

