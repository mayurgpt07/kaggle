from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


train_data = pd.read_csv("./data/kc_house_data.csv", sep = ",", header = 0)

#Start PreProcessing 
train_data['dateInFormat'] = train_data['date'].apply(lambda x: x[0:8])
train_data['dateInFormat'] = train_data['dateInFormat'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))
train_data['YearSold'] = train_data['dateInFormat'].apply(lambda x: int(x.year))
train_data['HomeAgeinYear'] = train_data['YearSold'] - train_data['yr_built']
train_data['RenovatedafterYears'] = train_data['yr_built'] - train_data['yr_renovated']
train_data['WaterFrontPresent'] = train_data['waterfront'].apply(lambda x: 1 if x == 1 else -1)
#Check the correlation matrix with price

#print(train_data['WaterFrontPresent'])
train_data.corr().loc[:,'id':'price'].style.background_gradient(cmap='coolwarm', axis=None)

#pd.plotting.scatter_matrix(train_data.loc[:,'price':'floors'], alpha = 0.2, figsize = (20, 20), diagonal = 'kde')
#plt.show()

train_data['Log_sqftlot'] = train_data['sqft_lot'].apply(lambda x: np.log(x))
train_data['Log_price'] = train_data['price'].apply(lambda x: np.log(x))
train_data['Log_sqftlot15'] = train_data['sqft_lot15'].apply(lambda x: np.log(x))
train_data['IsBasementThere'] = train_data['sqft_above'].apply(lambda x: 1 if x >= 1 else -1)
#plt.hist(train_data['sqft_lot'], color = "red")
#plt.hist(train_data['sqft_lot'], color = "skyblue")
#plt.show()

columnsToTrain = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition', 'grade', 'HomeAgeinYear', 'RenovatedafterYears','IsBasementThere',
       'lat', 'long','Log_sqftlot']


X, y = train_data.loc[:, columnsToTrain], train_data.loc[:, 'Log_price']

vif = pd.DataFrame()
New_X = add_constant(X)
vif['VIF Factors']  = [variance_inflation_factor(New_X.values, i) for i in range(New_X.shape[1])]
vif['Columns'] = New_X.columns
print(vif)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.30)


polynomialVariable = PolynomialFeatures(degree = 3)
polynomialCurveFitting = polynomialVariable.fit_transform(X_train)

polynomialVariable.fit(X_train, Y_train)



model = LinearRegression()

fittedModel = model.fit(X_train, Y_train)

model2 = LinearRegression()
fittedModel2 = model2.fit(polynomialCurveFitting, Y_train)

print(fittedModel.score(X_train, Y_train))
print(fittedModel.coef_)

print(fittedModel2.score(polynomialCurveFitting, Y_train))
print(fittedModel2.coef_)

formuala = 'Log_price ~ bedrooms+bathrooms+sqft_living+Log_sqftlot+floors+waterfront+view+condition+grade+HomeAgeinYear+RenovatedafterYears+IsBasementThere+lat+long'
statisticalModel = sm.ols(formuala, data = train_data)
statsfitted = statisticalModel.fit()
print(statsfitted.summary())

#print(fittedModel.predict(X_test))
