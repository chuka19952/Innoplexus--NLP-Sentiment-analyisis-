# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import Series,DataFrame 
from datetime import date
import datetime
import io
from scipy import stats
import re
import nltk
from nltk.tokenize import word_tokenize as WordTokenizer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import re
from nltk.corpus import stopwords
import pandas as pd

# Importing the dataset
train_set = pd.read_csv('train_F3WbcTw.csv')
sample_sub=pd.read_csv('sample_submission_i5xnIZD.csv')
test_set=pd.read_csv('test.csv')

#Creating and storing our target variable in a seperate dataframe
sentimentfrm=DataFrame(train_set["sentiment"])

#Here we want to merge our training and test set but for the columns to be same, we must first take of our target in train
train_set=train_set.drop(['sentiment'], axis=1)

#Now we merge BUT first we must create 2 train columns in our test and train set in order to id and seperate them later
train_set['train']=1
test_set['train']=0

combined=pd.concat([train_set, test_set])

"""One very important to note over here is that we haven’t specified the axis while concatenating which means we are combining along the rows.So what this will do is combine the test set below the train set with the ‘train’ column acting as the demarkation(all rows with 1 belong to train set and those with 0 to the test part).

Now do the encoding you require on the required column and save it in a new dataset."""


#Now we make our drug column into dummies as they turn out to have 2923 unique items/drugs
#NOW WE ENCODE
combined1=pd.get_dummies(combined['drug'], drop_first=True)

combined=combined.drop(['drug'], axis=1)

#combined=combined.drop(['unique_hash'], axis=1)
#combined=combined.drop(['train'], axis=1)

#removing any empty row or space
#combined['text'].dropna(inplace=True)

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split 
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stopword_set]

    # join the cleaned words in a list
    cleaned_word_list = " ".join(meaningful_words)

    return cleaned_word_list

def process_data(dataset):
    #tweets_df = pd.read_csv(dataset,delimiter='|',header=None)

    num_tweets = combined.shape[0]
    print("Total tweets: " + str(num_tweets))

    cleaned_tweets = []
    print("Beginning processing of tweets at: ")

    for i in range(num_tweets):
        cleaned_tweet = preprocess(combined.iloc[i][1])
        cleaned_tweets.append(cleaned_tweet)
        if(i % 8200 == 0):
            print(str(i) + " tweets processed")

    print("Finished processing of tweets at: ")
    return cleaned_tweets

    print("Finished processing of tweets at: ")
    return cleaned_tweets

cleaned_data = process_data(combined["text"])


cleaned_data=DataFrame(cleaned_data)

combined['text']=cleaned_data

#lemmitize the 'combined' dataframe
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
  return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

combined['text'] = combined.text.apply(lemmatize_text)


#combined2= Series(combined.text)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4000)
combined3 = cv.fit_transform(cleaned_data).toarray()
combined3=DataFrame(combined3)

#RESET index before concating else it would bring error
combined1.reset_index(drop=True, inplace=True)
combined3.reset_index(drop=True, inplace=True)
combined4 = pd.concat([combined3, combined1], axis=1)

combined4.reset_index(drop=True, inplace=True)
combined.reset_index(drop=True, inplace=True)
combined=combined.drop(['text'], axis=1)
combined=pd.concat([combined4, combined], axis=1)
