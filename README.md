## Kaggle Problems

#### Titanic Survivor
File Name - Titanic. py <br />
The problem is the first problem on kaggle. Using logistic regression I have tried to come up with a probable solution for it.
The above code gave me a score of about 80.38% on kaggle whereas the cross validation score for a split in data set of ratio 70:30 was 83.26%.

#### Kings County Housing Price
File Name - KC_House_Data.py <br />
The problem is based on predicting house prices in the King's County region. In the problem, I have tried using multiple regression techniques like Least Square Linear Regression Technique (an adjusted R^2 value of 80.23%), Polynomial Regression (10-fold cross validation adjusted R^2 value of 85%) and Ridge Regression (10-fold cross validation value of 84.5%). The impressive thing is that even though the Polynomial regression has a better cross validation score than Ridge Regression, but due to the tunning parameter(lambda or alpha in scikit model) and the shrinkage penality we are able to give a better prediction.

#### Toxic Comments Classification
File Name - NaturalToxicLanguage.py <br />
It is a Natural Language Processing and Classification problem, that involves specifying whether a comment is Toxic, Severe Toxic, Obscene, Threat, Insult and Identity Hate. In the problem, I am using TFIDF algorithm for word embeddings and then Logistic Regression, Support Vector classifier to predict which class a comment belongs to.  

