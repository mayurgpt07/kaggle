from sklearn import linear_model
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, KFold, cross_val_score

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
imputer = preprocessing.Imputer(missing_values = 0, strategy = 'mean')
imputed_train_data = train_data
imputed_test_data = test_data
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }
imputed_train_data['Age'] = imputed_train_data['Age'].replace(np.NaN,0.0)
imputed_train_data['Age'] = imputer.fit_transform(imputed_train_data['Age'].values.reshape(-1,1))
imputed_train_data['Age'] = np.round(imputed_train_data['Age'])
imputed_train_data['Age'] = np.fix(imputed_train_data['Age'])
imputed_train_data['Age'] = imputed_train_data['Age'].astype(int)

imputed_test_data['Age'] = imputed_test_data['Age'].replace(np.NaN,0.0)
imputed_test_data['Age'] = np.round(imputed_test_data['Age'])
imputed_test_data['Age'] = np.fix(imputed_test_data['Age'])
imputed_test_data['Age'] = imputed_test_data['Age'].astype(int)
imputed_test_data['Age'] = imputer.fit_transform(imputed_test_data['Age'].values.reshape(-1,1))


imputed_test_data['Fare'] = imputed_test_data['Fare'].replace(np.NaN,0.0)
imputed_test_data['Fare'] = imputer.fit_transform(imputed_test_data['Fare'].values.reshape(-1,1))

imputed_train_data = imputed_train_data.join(pd.get_dummies(imputed_train_data['Embarked']))
imputed_train_data = imputed_train_data.join(pd.get_dummies(imputed_train_data['Sex']))
imputed_train_data["Age"] = imputed_train_data["Age"]/imputed_train_data["Age"].max()
imputed_train_data["Fare"] = imputed_train_data["Fare"]/imputed_train_data["Fare"].max()
imputed_train_data["CabinAvailable"] = np.where(imputed_train_data['Cabin'].isnull(),'No','Yes')
imputed_train_data = imputed_train_data.join(pd.get_dummies(imputed_train_data['CabinAvailable']))
imputed_train_data["FamilySize"] = imputed_train_data["SibSp"] + imputed_train_data["Parch"] + 1
imputed_train_data['FamilyGroup'] = np.where(imputed_train_data['FamilySize'] == 1, 'Alone',np.where(imputed_train_data['FamilySize'] >=5,'Big','Small'))
imputed_train_data = imputed_train_data.join(pd.get_dummies(imputed_train_data['FamilyGroup']))
imputed_train_data = imputed_train_data.join(pd.get_dummies(imputed_train_data["Pclass"]))
imputed_train_data['Title'] = imputed_train_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
imputed_train_data['Title'] = imputed_train_data.Title.map(Title_Dictionary)
imputed_train_data = imputed_train_data.join(pd.get_dummies(imputed_train_data['Title']))
# print(len(imputed_train_data['Cabin'][0].split(" ")))

# print(len(imputed_train_data['Cabin'][0].strip().split(" ")))
# print(imputed_train_data.loc['CabinNumber',:])
# imputed_test_data['BinarySex'l] = np.where(imputed_test_data['Sex'] == 'male', 1, 0)
imputed_test_data = imputed_test_data.join(pd.get_dummies(imputed_test_data['Embarked']))
imputed_test_data = imputed_test_data.join(pd.get_dummies(imputed_test_data['Sex']))
imputed_test_data["Age"] = imputed_test_data["Age"]/imputed_test_data["Age"].max()
imputed_test_data["Fare"] = imputed_test_data["Fare"]/imputed_test_data["Fare"].max()
imputed_test_data["CabinAvailable"] = np.where(imputed_test_data['Cabin'].isnull(),'No','Yes')
imputed_test_data = imputed_test_data.join(pd.get_dummies(imputed_test_data['CabinAvailable']))
imputed_test_data["FamilySize"] = imputed_test_data["SibSp"] + imputed_test_data["Parch"] + 1
imputed_test_data['FamilyGroup'] = np.where(imputed_test_data['FamilySize'] == 1, 'Alone',np.where(imputed_test_data['FamilySize'] >=5,'Big','Small'))
imputed_test_data = imputed_test_data.join(pd.get_dummies(imputed_train_data['FamilyGroup']))
imputed_test_data = imputed_test_data.join(pd.get_dummies(imputed_test_data["Pclass"]))
imputed_test_data['Title'] = imputed_test_data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
imputed_test_data['Title'] = imputed_test_data.Title.map(Title_Dictionary)
imputed_test_data = imputed_test_data.join(pd.get_dummies(imputed_test_data['Title']))

imputed_train_data['Cabin'] = np.where(imputed_train_data['Cabin'].isnull(),'No',imputed_train_data['Cabin'])
for i, row in imputed_train_data.iterrows():
    if(row['Cabin'] == 'No'):
        imputed_train_data.set_value(i,'CabinNumber',int(0))
    else:
        imputed_train_data.set_value(i,'CabinNumber',int(len(imputed_train_data['Cabin'][i].split(" "))))
imputed_train_data['CabinNumber'] = np.round(imputed_train_data['CabinNumber'])
imputed_train_data['CabinNumber'] = np.fix(imputed_train_data['CabinNumber'])
imputed_train_data['CabinNumber'] = imputed_train_data['CabinNumber'].astype(int)
imputed_train_data = imputed_train_data.join(pd.get_dummies(imputed_train_data['CabinNumber'], prefix='Cabin'))

imputed_test_data['Cabin'] = np.where(imputed_test_data['Cabin'].isnull(),'No',imputed_test_data['Cabin'])
for i, row in imputed_test_data.iterrows():
    if(row['Cabin'] == 'No'):
        imputed_test_data.set_value(i,'CabinNumber',int(0))
    else:
        imputed_test_data.set_value(i,'CabinNumber',int(len(imputed_test_data['Cabin'][i].split(" "))))


imputed_test_data['CabinNumber'] = np.round(imputed_test_data['CabinNumber'])
imputed_test_data['CabinNumber'] = np.fix(imputed_test_data['CabinNumber'])
imputed_test_data['CabinNumber'] = imputed_test_data['CabinNumber'].astype(int)
imputed_test_data = imputed_test_data.join(pd.get_dummies(imputed_test_data['CabinNumber'],prefix='Cabin'))

print(imputed_train_data.iloc[1:3,35:42])
print(imputed_train_data.iloc[1:3,12:20])
classifier = ExtraTreesClassifier()
# print(imputed_train_data.iloc[:,1])                    0           0                 0  0             0
classifier.fit(imputed_train_data.iloc[:,[5,15,16,18,19,22,23,24,25,26,27,29,30,31,32,33,34,36,37,38,39,40]], imputed_train_data.iloc[:,1])
print(classifier.feature_importances_)
# print(imputed_train_data.iloc[1:3, 20:28])
#                                            0               0        0        0        0
X, y = imputed_train_data.iloc[:, [5,6,15,16,18,19,23,24,25,26,27,29,30,31,32,36,37,38,39]], imputed_train_data.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

logistic_model = linear_model.LogisticRegression(max_iter = 100)
logistic_model.fit(X, y)
print(accuracy_score(logistic_model.predict(X),y))
logistic_model2 = linear_model.LogisticRegression(max_iter = 100)
logistic_model2.fit(X_train, y_train)
print(accuracy_score(logistic_model2.predict(X_test), y_test))
# print(np.std(X, 0)*logistic_model.coef_[0])

fpr, tpr, thresholds = roc_curve(y_test, logistic_model2.predict(X_test))
auc_score = auc(fpr, tpr)

print(auc_score)

Kfold = KFold(n_splits=10, random_state=7)
scoring = 'roc_auc'
res = cross_val_score(logistic_model2, X, y, cv = Kfold, scoring = scoring)
# print(res.mean(), res.std())
plt.title("ROC")
plt.plot(fpr, tpr, 'b')
# plt.show()

print(np.std(X_train, 0)*logistic_model2.coef_[0])
# print(logistic_model2.score(X_test, y_test))
print(imputed_test_data.iloc[1:4,24:27])
to_test = imputed_test_data.iloc[:, [4,6,14,15,17,18,22,23,24,25,26,28,29,30,31,35,36,37,38]]
results = logistic_model2.predict(to_test)

d = {"PassengerId": pd.Series(imputed_test_data.iloc[:,0]), "Survived": pd.Series(results)}

result_dataframe = pd.DataFrame(d)
result_dataframe.to_csv("/home/mayur/workspace/learn/gradual/Result.csv", header = True)
# # print(imputed_test_data.iloc[:,0],logistic_model2.predict(to_test))
# print(accuracy_score(logistic_model.predict(to_test),logistic_model2.predict(to_test)))
# classifier = ExtraTreesClassifier()
# classifier.fit(X, y)
# print(classifier.feature_importances_)
#
# model = SelectFromModel(classifier, prefit = True)
# model.transform(X)
# print(model.transform(X).shape)
