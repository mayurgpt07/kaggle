import pandas as pd
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pattern.en import suggest
import re

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
train_data['comment_text'].fillna('unknown', inplace = True)
#test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\s+',' ',x.strip().lower()))
#test_data['new_Comment_Text'] = test_data['comment_text'].apply(lambda x:re.sub('\s+',' ',x.strip().lower()))

train_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', x))
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
train_data['RemovedStopWords'] = train_data['RemovedStopWords'].apply(lambda x: re.sub('\s+', ' ', x.strip()))

toxic_data = pd.DataFrame()
Severetoxic_data = pd.DataFrame()
Obscenetoxic_data = pd.DataFrame()
Threattoxic_data = pd.DataFrame()
Insulttoxic_data = pd.DataFrame()
Identitytoxic_data = pd.DataFrame()

toxic_data['Comments'] = train_data[train_data['toxic'] == 1]['RemovedStopWords']
toxic_data['lengthOfLength'] = toxic_data['Comments'].apply(lambda x: len(x.strip()))
toxic_data['numberOfWords'] = toxic_data['Comments'].apply(lambda x: len(x.split()))
plt.hist(toxic_data['lengthOfLength'], color = "red")
plt.show()

print(toxic_data.head())

Severetoxic_data['Comments'] = train_data[train_data['severe_toxic'] == 1]['RemovedStopWords']
Severetoxic_data['lengthOfLength'] = Severetoxic_data['Comments'].apply(lambda x: len(x.strip()))
Severetoxic_data['numberOfWords'] = Severetoxic_data['Comments'].apply(lambda x: len(x.split()))
plt.hist(Severetoxic_data['lengthOfLength'], color = "red")
plt.show()

Obscenetoxic_data['Comments'] = train_data[train_data['obscene'] == 1]['RemovedStopWords']
Obscenetoxic_data['lengthOfLength'] = Obscenetoxic_data['Comments'].apply(lambda x: len(x.strip()))
Obscenetoxic_data['numberOfWords'] = Obscenetoxic_data['Comments'].apply(lambda x: len(x.split()))
plt.hist(Obscenetoxic_data['lengthOfLength'], color = "red")
plt.show()


Threattoxic_data['Comments'] = train_data[train_data['threat'] == 1]['RemovedStopWords']
Threattoxic_data['lengthOfLength'] = Threattoxic_data['Comments'].apply(lambda x: len(x.strip()))
Threattoxic_data['numberOfWords'] = Threattoxic_data['Comments'].apply(lambda x: len(x.split()))
plt.hist(Threattoxic_data['lengthOfLength'], color = "red")
plt.show()


Insulttoxic_data['Comments'] = train_data[train_data['insult'] == 1]['RemovedStopWords']
Insulttoxic_data['lengthOfLength'] = Insulttoxic_data['Comments'].apply(lambda x: len(x.strip()))
Insulttoxic_data['numberOfWords'] = Insulttoxic_data['Comments'].apply(lambda x: len(x.split()))
plt.hist(Insulttoxic_data['lengthOfLength'], color = "red")
plt.show()


Identitytoxic_data['Comments'] = train_data[train_data['identity_hate'] == 1]['RemovedStopWords']
Identitytoxic_data['lengthOfLength'] = Identitytoxic_data['Comments'].apply(lambda x: len(x.strip()))
Identitytoxic_data['numberOfWords'] = Identitytoxic_data['Comments'].apply(lambda x: len(x.split()))
plt.hist(Identitytoxic_data['lengthOfLength'], color = "red")
plt.show()


empty_string = ''
for i in toxic_data['Comments']:
	empty_string = empty_string.strip() + ' ' + i.strip()

empty_string1 = ''
for i in Severetoxic_data['Comments']:
	empty_string1 = empty_string1.strip() + ' ' + i.strip()

empty_string2 = ''
for i in Obscenetoxic_data['Comments']:
	empty_string2 = empty_string2.strip() + ' ' + i.strip()

empty_string3 = ''
for i in Threattoxic_data['Comments']:
	empty_string3 = empty_string3.strip() + ' ' + i.strip()

empty_string4 = ''
for i in Insulttoxic_data['Comments']:
	empty_string4 = empty_string4.strip() + ' ' + i.strip()

empty_string5 = ''
for i in Identitytoxic_data['Comments']:
	empty_string5 = empty_string5.strip() + ' ' + i.strip()


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