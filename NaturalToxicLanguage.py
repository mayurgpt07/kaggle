import pandas as pd
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
#test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\s+',' ',x.strip().lower()))
#test_data['new_Comment_Text'] = test_data['comment_text'].apply(lambda x:re.sub('\s+',' ',x.strip().lower()))

train_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,',' ', x))
#test_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,',' ', x))
 
train_data['CommentTokenize'] = train_data['new_Comment_Text'].apply(lambda x: word_tokenize(x))
#test_data['CommentTokenize'] = test_data['new_Comment_Text'].apply(lambda x:word_tokenize(x))

sentenceList = []
sentenceListTest = []

for i in train_data['CommentTokenize']:
	wordList = ''	
	for k in i:
		if lemmatizer.lemmatize(k) not in stopwordsList:
			wordList = wordList + reduce_lengthening(lemmatizer.lemmatize(k)) + ' '
	sentenceList.append(wordList.strip())
#wordList.clear()

#for i in test_data['CommentTokenize']:
#	wordListTest = ''
#	for k in i:
#		if k not in stopwordsList:
#			wordListTest = wordListTest + k + ' '
#	sentenceListTest.append(wordListTest.lstrip())

train_data['RemovedStopWords'] = pd.Series(sentenceList)

toxic_data = train_data[train_data['toxic'] == 1]['RemovedStopWords']

Severetoxic_data = train_data[train_data['severe_toxic'] == 1]['RemovedStopWords']

Obscenetoxic_data = train_data[train_data['obscene'] == 1]['RemovedStopWords']

Threattoxic_data = train_data[train_data['threat'] == 1]['RemovedStopWords']

Insulttoxic_data = train_data[train_data['insult'] == 1]['RemovedStopWords']

Identitytoxic_data = train_data[train_data['identity_hate'] == 1]['RemovedStopWords']

empty_string = ''
for i in toxic_data:
	empty_string = empty_string + ' ' + i

empty_string1 = ''
for i in Severetoxic_data:
	empty_string1 = empty_string1 + ' ' + i

empty_string2 = ''
for i in Obscenetoxic_data:
	empty_string2 = empty_string2 + ' ' + i

empty_string3 = ''
for i in Threattoxic_data:
	empty_string3 = empty_string3 + ' ' + i

empty_string4 = ''
for i in Insulttoxic_data:
	empty_string4 = empty_string4 + ' ' + i

empty_string5 = ''
for i in Identitytoxic_data:
	empty_string5 = empty_string5 + ' ' + i

wordcloud = WordCloud(width = 900, height = 900, 
                background_color ='white',
                min_font_size = 10).generate(empty_string) 
 
wordcloud1 = WordCloud(width = 900, height = 900, 
                background_color ='white',
                min_font_size = 10).generate(empty_string1)

wordcloud2 = WordCloud(width = 900, height = 900, 
                background_color ='white',
                min_font_size = 10).generate(empty_string2)

wordcloud3 = WordCloud(width = 900, height = 900, 
                background_color ='white',
                min_font_size = 10).generate(empty_string3)

wordcloud4 = WordCloud(width = 900, height = 900, 
                background_color ='white',
                min_font_size = 10).generate(empty_string4)

wordcloud5 = WordCloud(width = 900, height = 900, 
                background_color ='white',
                min_font_size = 10).generate(empty_string5)                                                                  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud1)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud2)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud3)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud4)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()


plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud5)
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show() 

#vectorizer = TfidfVectorizer(min_df = 0.05, strip_accents = 'unicode', use_idf = 1, sublinear_tf = 1)
#X = vectorizer.fit_transform(train_data['RemovedStopWords'])

#print(vectorizer.get_feature_names())
#print(X.shape)

#test_data['RemovedStopWords'] = pd.Series(sentenceListTest)
print(train_data.head())