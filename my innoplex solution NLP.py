# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas import Series,DataFrame 
from datetime import date
import datetime as DT
import io
from scipy import stats


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

# Cleaning the texts
import re
import nltk
from nltk.tokenize import word_tokenize as WordTokenizer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 8202):
    review = re.sub('[^a-zA-Z]',' ', combined['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)



combined['text']=re.sub('[^a-zA-Z]', ' ', combined['text'])
combined['text']=combined['text'].lower()
combined["text"] = combined["text"].apply(nltk.word_tokenize)
ps = PorterStemmer()
combined['text'] = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
combined['text'] = ' '.join(combined['text'])