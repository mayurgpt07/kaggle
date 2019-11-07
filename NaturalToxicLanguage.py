import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from spellchecker import SpellChecker

train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
spellingCheck = SpellChecker()

train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\s+',' ',x.strip().lower()))
test_data['new_Comment_Text'] = test_data['comment_text'].apply(lambda x:re.sub('\s+',' ',x.strip().lower()))

train_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,',' ', x))
test_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,',' ', x))
 
train_data['CommentTokenize'] = train_data['new_Comment_Text'].apply(lambda x: word_tokenize(x))
test_data['CommentTokenize'] = test_data['new_Comment_Text'].apply(lambda x:word_tokenize(x))

sentenceList = []
sentenceListTest = []

for i in train_data['CommentTokenize']:
	wordList = ''	
	for k in i:
		if k not in stopwordsList:
			wordList = wordList + lemmatizer.lemmatize(spellingCheck.correction(k)) + ' '
	sentenceList.append(wordList.strip())
#wordList.clear()

for i in test_data['CommentTokenize']:
	wordListTest = ''
	for k in i:
		if k not in stopwordsList:
			wordListTest = wordListTest + lemmatizer.lemmatize(spellingCheck.correction(k)) + ' '
	sentenceListTest.append(wordListTest.lstrip())

train_data['RemovedStopWords'] = pd.Series(sentenceList)

vectorizer = TfidfVectorizer(strip_accents = 'unicode', use_idf = 1, sublinear_tf = 1)
X = vectorizer.fit_transform(train_data['RemovedStopWords'])

print(vectorizer.get_feature_names())
print(X.shape)

test_data['RemovedStopWords'] = pd.Series(sentenceListTest)
print(train_data.head())