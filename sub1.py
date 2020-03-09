import pandas as pd
import numpy as np

dataset = pd.read_csv('train_2kmZucJ.csv')
testset=pd.read_csv('test_oJQbWVk.csv')
# Cleaning the texts
nltk.download('wordnet')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus = []
corp=[]

for i in range(0, 7920):
    tweet = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = WordNetLemmatizer()
    tweet = [ps.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

for i in range(0, 1953):
    tweet = re.sub('[^a-zA-Z]', ' ', testset['tweet'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = WordNetLemmatizer()
    tweet = [ps.lemmatize(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corp.append(tweet)
    
#bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer(max_features=5000)
X1 = cv1.fit_transform(corp).toarray()


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X, y)

y_pred = classifier.predict(X1)


