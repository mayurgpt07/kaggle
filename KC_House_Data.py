from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.formula.api as sm


train_data = pd.read_csv("./data/kc_house_data.csv", sep = ",", header = 0)

#Start PreProcessing 
train_data['dateInFormat'] = train_data['date'].apply(lambda x: x[0:8])
train_data['dateInFormat'] = train_data['dateInFormat'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))

print(train_data)
#Check the correlation matrix with price
train_data.corr().loc[:,'id':'price'].style.background_gradient(cmap='coolwarm', axis=None)

#pd.plotting.scatter_matrix(train_data.loc[:,'price':'floors'], alpha = 0.2, figsize = (20, 20), diagonal = 'kde')
#plt.show()

X, y = train_data.loc[:, 'bedrooms': 'sqft_lot15'], train_data.loc[:, 'price']

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.33)

train_data['Log_sqftlot'] = train_data['sqft_lot'].apply(lambda x: np.log(x))
train_data['Log_price'] = train_data['price'].apply(lambda x: np.log(x))
#plt.hist(train_data['sqft_lot'], color = "red")
plt.hist(train_data['sqft_lot'], color = "skyblue")
plt.hist(train_data['price'], color = 'red')
plt.show()

#model = LinearRegression()

#fittedModel = model.fit(X_train, Y_train)

#print(fittedModel.score(X_train, Y_train))
#print(fittedModel.coef_)


#formuala = 'price ~ bedrooms+bathrooms+sqft_living+sqft_lot+floors+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+zipcode+lat+long+sqft_living15+sqft_lot15'
#statisticalModel = sm.ols(formuala, data = train_data)
#statsfitted = statisticalModel.fit()
#print(statsfitted.summary())

#print(fittedModel.predict(X_test))