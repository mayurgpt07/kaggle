import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
import re

train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))

train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\s+',' ',x.strip()))
test_data['new_Comment_Text'] = test_data['comment_text'].apply(lambda x:re.sub('\s+',' ',x.strip()))

train_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\?|\.|\!|\/|\;|\:|\=|\"|\:',' ', x))
test_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\?|\.|\!|\/|\;|\:|\=|\"|\:',' ', x))
 
train_data['CommentTokenize'] = train_data['new_Comment_Text'].apply(lambda x: word_tokenize(x))
test_data['CommentTokenize'] = test_data['new_Comment_Text'].apply(lambda x:word_tokenize(x))

sentenceList = []


for i in train_data['CommentTokenize']:
	wordList = ''	
	for k in i:
		if k not in stopwordsList:
			wordList = wordList + ' '+k
	sentenceList.append(wordList.strip())
#wordList.clear()

for i in test_data['CommentTokenize']:
	wordListTest = ''
	for k in i:
		if k not in stopwordsList:
			wordListTest = wordListTest + ' ' + k
	sentenceListTest.append(wordListTest.strip())

train_data['RemovedStopWords'] = pd.Series(sentenceList)
test_data['RemovedStopWords'] = pd.Series(sentenceListTest)
print(train_data.head())