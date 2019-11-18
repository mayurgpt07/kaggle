import pandas as pd
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from sklearn import model_selection
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from pattern.en import suggest
import re
from gensim.models import Word2Vec


def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

'''def createFeatures(train_data):
	toxic_data = pd.DataFrame()
	toxic_data['Comments'] = train_data.loc[:,'new_Comment_Text']
	print(toxic_data['Comments'].head(10))
	toxic_data['ClassResult'] = train_data.loc[:, 'toxic']
	toxic_data['NumberOfSentences'] = toxic_data['Comments'].apply(lambda x: len(re.findall('\n',str(x.strip())))+1)
	toxic_data['MeanLengthOfSentences'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split("\n")]))
	toxic_data['MeanLengthOfWords'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split(" ")]))
	toxic_data['NumberOfUniqueWords'] = toxic_data['Comments'].apply(lambda x: len(set(x.split())))
	toxic_data['numberOfWords'] = toxic_data['Comments'].apply(lambda x: len(x.split()))
	return toxic_data'''



train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
train_data['comment_text'].fillna('unknown', inplace = True)
#test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\s+',' ',x.strip().lower()))
#test_data['new_Comment_Text'] = test_data['comment_text'].apply(lambda x:re.sub('\s+',' ',x.strip().lower()))

train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', x))
#test_data['new_Comment_Text'] = train_data['new_Comment_Text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,',' ', x))
 
#train_data['CommentTokenize'] = train_data['new_Comment_Text'].apply(lambda x: word_tokenize(x))
#test_data['CommentTokenize'] = test_data['new_Comment_Text'].apply(lambda x:word_tokenize(x))


'''for i in train_data['CommentTokenize']:
	wordList = ''	
	for k in i:
		if lemmatizer.lemmatize(k) not in stopwordsList:
			wordList = wordList + reduce_lengthening(lemmatizer.lemmatize(k)) + ' '
	sentenceList.append(wordList.strip().lower())'''
#wordList.clear()

#for i in test_data['CommentTokenize']:
#	wordListTest = ''
#	for k in i:
#		if k not in stopwordsList:
#			wordListTest = wordListTest + k + ' '
#	sentenceListTest.append(wordListTest.lstrip())

#train_data['RemovedStopWords'] = pd.Series(sentenceList)
#train_data['RemovedStopWords'] = train_data['RemovedStopWords'].apply(lambda x: re.sub('\s+', ' ', x.strip()))

#vectorizer = TfidfVectorizer(min_df = 150, strip_accents = 'unicode', ngram_range = (1,4), stop_words = 'english', sublinear_tf = True)
#Y = vectorizer.fit_transform(train_data['new_Comment_Text'])
#print(vectorizer.get_feature_names())

#toxic_data = pd.DataFrame()
'''Severetoxic_data = pd.DataFrame()
Obscenetoxic_data = pd.DataFrame()
Threattoxic_data = pd.DataFrame()
Insulttoxic_data = pd.DataFrame()
Identitytoxic_data = pd.DataFrame()'''

'''toxic_data['Comments'] = train_data.loc[:,'RemovedStopWords']
toxic_data['Comments'].head(10)
toxic_data['ClassResult'] = train_data[train_data['toxic'] == 1]['toxic']
toxic_data['NumberOfSentences'] = toxic_data['Comments'].apply(lambda x: len(re.findall("\n",str(x.strip())))+1)
toxic_data['MeanLengthOfSentences'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split("\n")]))
toxic_data['MeanLengthOfWords'] = toxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.strip().split(" ")]))
toxic_data['NumberOfUniqueWords'] = toxic_data['Comments'].apply(lambda x: len(set(x.split())))
toxic_data['numberOfWords'] = toxic_data['Comments'].apply(lambda x: len(x.split()))'''

#toxic_data.to_csv('IntermediateDataFrame.csv', sep = ',', header = True)
toxic_data_pandas = pd.read_csv('./IntermediateDataFrame.csv', sep = ',', header = 0)

toxic_data = cudf.from_pandas(toxic_data_pandas)
toxic_data['Comment_Text'] = toxic_data['Comments'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', x))

toxic_data['new_Comment_Text'] = toxic_data['Comment_Text'].apply(lambda x: re.sub('\s+',' ',x.strip().lower()))

toxic_data['CommentTokenize'] = toxic_data['new_Comment_Text'].apply(lambda x: word_tokenize(x))


sentenceList = []
sentenceListTest = []

for i in toxic_data['CommentTokenize']:
	wordList = ''	
	for k in i:
		if lemmatizer.lemmatize(k) not in stopwordsList:
			wordList = wordList + reduce_lengthening(lemmatizer.lemmatize(k)) + ' '
	sentenceList.append(wordList.strip().lower())
#wordList.clear()

toxic_data['RemovedStopWords'] = pd.Series(sentenceList)
toxic_data['RemovedStopWords'] = toxic_data['RemovedStopWords'].apply(lambda x: re.sub('\s+', ' ', x.strip()))


#toxic_data.to_csv('IntermediateDataFrame.csv', sep = ',', header = True)
'''print(toxic_data.head())

Severetoxic_data['Comments'] = train_data[train_data['severe_toxic'] == 1]['RemovedStopWords']
Severetoxic_data['NumberOfSentences'] = Severetoxic_data['Comments'].apply(lambda x: len(re.findall("\n",str(x)))+1)
Severetoxic_data['MeanLengthOf Sentences'] = Severetoxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.split()]))
Severetoxic_data['NumberOfUniqueWords'] = Severetoxic_data['Comments'].apply(lambda x: len(set(x.split())))
Severetoxic_data['numberOfWords'] = Severetoxic_data['Comments'].apply(lambda x: len(x.split()))

Obscenetoxic_data['Comments'] = train_data[train_data['obscene'] == 1]['RemovedStopWords']
Obscenetoxic_data['NumberOfSentences'] = Obscenetoxic_data['Comments'].apply(lambda x: len(re.findall("\n",str(x)))+1)
Obscenetoxic_data['MeanLengthOf Sentences'] = Obscenetoxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.split()]))
Obscenetoxic_data['NumberOfUniqueWords'] = Obscenetoxic_data['Comments'].apply(lambda x: len(set(x.split())))
Obscenetoxic_data['numberOfWords'] = Obscenetoxic_data['Comments'].apply(lambda x: len(x.split()))



Threattoxic_data['Comments'] = train_data[train_data['threat'] == 1]['RemovedStopWords']
Threattoxic_data['NumberOfSentences'] = Threattoxic_data['Comments'].apply(lambda x: len(re.findall("\n",str(x)))+1)
Threattoxic_data['MeanLengthOf Sentences'] = Threattoxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.split()]))
Threattoxic_data['NumberOfUniqueWords'] = Threattoxic_data['Comments'].apply(lambda x: len(set(x.split())))
Threattoxic_data['numberOfWords'] = Threattoxic_data['Comments'].apply(lambda x: len(x.split()))


Insulttoxic_data['Comments'] = train_data[train_data['insult'] == 1]['RemovedStopWords']
Insulttoxic_data['NumberOfSentences'] = Insulttoxic_data['Comments'].apply(lambda x: len(re.findall("\n",str(x)))+1)
Insulttoxic_data['MeanLengthOf Sentences'] = Insulttoxic_data1['Comments'].apply(lambda x: np.mean([len(w) for w in x.split()]))
Insulttoxic_data['NumberOfUniqueWords'] = Insulttoxic_data['Comments'].apply(lambda x: len(set(x.split())))
Insulttoxic_data['numberOfWords'] = Insulttoxic_data['Comments'].apply(lambda x: len(x.split()))


Identitytoxic_data['Comments'] = train_data[train_data['identity_hate'] == 1]['RemovedStopWords']
Identitytoxic_data['NumberOfSentences'] = Identitytoxic_data['Comments'].apply(lambda x: len(re.findall("\n",str(x)))+1)
Identitytoxic_data['MeanLengthOf Sentences'] = Identitytoxic_data['Comments'].apply(lambda x: np.mean([len(w) for w in x.split()]))
Identitytoxic_data['NumberOfUniqueWords'] = Identitytoxic_data['Comments'].apply(lambda x: len(set(x.split())))
Identitytoxic_data['numberOfWords'] = Identitytoxic_data['Comments'].apply(lambda x: len(x.split()))
'''


empty_string = ''
for i in toxic_data['RemovedStopWords']:
	empty_string = empty_string.strip() + ' ' + i.strip()

print('Length of Total Comments', len(empty_string))
'''empty_string1 = ''
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
	empty_string5 = empty_string5.strip() + ' ' + i.strip()'''


wordcloud = WordCloud(width = 900, height = 900,
                background_color ='white',
                min_font_size = 10).generate(empty_string)

vectorizer = TfidfVectorizer(min_df = 180, strip_accents = 'unicode', ngram_range = (1,4), stop_words = 'english', sublinear_tf = True)
X = vectorizer.fit_transform(toxic_data['RemovedStopWords'])
print('Length of features', len(vectorizer.get_feature_names()))
train_ngrams = vectorizer.transform(toxic_data['RemovedStopWords'])

print(type(train_ngrams))
print(np.shape(train_ngrams))
print(np.ndim(train_ngrams))

trainingColumns = ['NumberOfSentences', 'NumberOfUniqueWords', 'numberOfWords', 'MeanLengthOfSentences']
testingColumns = ['ClassResult']


X, y = toxic_data.loc[:, trainingColumns], toxic_data.loc[:, testingColumns]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.33)

classifier = SVC(gamma = 'auto')
classifier.fit(X_train, Y_train)
#print(classifier.score())
'''wordcloud1 = WordCloud(width = 900, height = 900,
                background_color ='white',
                min_font_size = 10).generate(empty_string1)

vectorizer = TfidfVectorizer(min_df = 80, strip_accents = 'unicode', ngram_range = (1,4), stop_words = 'english', sublinear_tf = True)
X1 = vectorizer.fit_transform(Severetoxic_data['Comments'])
print(vectorizer.get_feature_names())

wordcloud2 = WordCloud(width = 900, height = 900,
                background_color ='white',
                min_font_size = 10).generate(empty_string2)

vectorizer = TfidfVectorizer(min_df = 180, strip_accents = 'unicode', ngram_range = (1,4), stop_words = 'english', sublinear_tf = True)
X2 = vectorizer.fit_transform(Obscenetoxic_data['Comments'])
print(vectorizer.get_feature_names())

wordcloud3 = WordCloud(width = 900, height = 900,
                background_color ='white',
                min_font_size = 10).generate(empty_string3)

vectorizer = TfidfVectorizer(min_df = 70, strip_accents = 'unicode', ngram_range = (1,4), stop_words = 'english', sublinear_tf = True)
X3 = vectorizer.fit_transform(Threattoxic_data['Comments'])
print(vectorizer.get_feature_names())

wordcloud4 = WordCloud(width = 900, height = 900,
                background_color ='white',
                min_font_size = 10).generate(empty_string4)

vectorizer = TfidfVectorizer(min_df = 180, strip_accents = 'unicode', ngram_range = (1,4), stop_words = 'english', sublinear_tf = True)
X4 = vectorizer.fit_transform(Insulttoxic_data['Comments'])
print(vectorizer.get_feature_names())

wordcloud5 = WordCloud(width = 900, height = 900,
                background_color ='white',
                min_font_size = 10).generate(empty_string5)

vectorizer = TfidfVectorizer(min_df = 80, strip_accents = 'unicode', ngram_range = (1,4), stop_words = 'english', sublinear_tf = True)
X5 = vectorizer.fit_transform(Identitytoxic_data['Comments'])
print(vectorizer.get_feature_names())'''

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()

'''plt.figure(figsize = (8, 8), facecolor = None)
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
print(train_data.head())'''