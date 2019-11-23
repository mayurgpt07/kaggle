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
from numba import jit

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def featureEngineer(toxic_data):
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
	return toxic_data

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



#train_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/train.csv', sep = ',', header = 0)
#train_data['comment_text'].fillna('unknown', inplace = True)
#test_data = pd.read_csv('./jigsaw-toxic-comment-classification-challenge/test.csv', sep = ',', header = 0)

stopwordsList = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\s+',' ',x.strip().lower()))
#test_data['new_Comment_Text'] = test_data['comment_text'].apply(lambda x:re.sub('\s+',' ',x.strip().lower()))

#train_data['new_Comment_Text'] = train_data['comment_text'].apply(lambda x: re.sub('\ |\!|\/|\;|\:|\=|\"|\:|\]|\[|\<|\>|\{|\}|\'|\?|\.|\,|\|',' ', x))
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
toxic_data_read = pd.read_csv('./IntermediateDataFrame.csv', sep = ',', header = 0)

toxic_data_Comments_Series = toxic_data_read['Comments'].to_numpy()
toxic_data = featureEngineer(toxic_data_read)


#toxic_data.to_csv('IntermediateDataFrame.csv', sep = ',', header = True)


empty_string = ''
for i in toxic_data['RemovedStopWords']:
	empty_string = empty_string.strip() + ' ' + i.strip()

print('Length of Total Comments', len(empty_string))

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


# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud)
plt.axis("off") 
plt.tight_layout(pad = 0)
plt.show()
