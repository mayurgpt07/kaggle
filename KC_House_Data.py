from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.kernel_approximation import Nystroem
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import math
import reverse_geocoder as rgc
import pprint as pp

train_data = pd.read_csv("./data/kc_house_data.csv", sep = ",", header = 0)

#Start PreProcessing
#Changing the date format to get the year 
train_data['dateInFormat'] = train_data['date'].apply(lambda x: x[0:8])
train_data['dateInFormat'] = train_data['dateInFormat'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))
#The year in which the propert was sold
train_data['YearSold'] = train_data['dateInFormat'].apply(lambda x: int(x.year))

#Since the value of Built Year and Selling Year is large, its good to normalize the value of different variables to similar range

#We derive the Time since its built
train_data['HomeAgeinYear'] = train_data['YearSold'] - train_data['yr_built']

#We derive the Time after which it was renovated
train_data['RenovatedafterYears'] = train_data['yr_renovated'] - train_data['yr_built']

#Creating WaterFront as a dummy variable
train_data['IsRenovated'] = train_data['waterfront'].apply(lambda x: 1 if x > 0 else -1)


#pd.plotting.scatter_matrix(train_data.loc[:,'price':'floors'], alpha = 0.2, figsize = (20, 20), diagonal = 'kde')
#plt.show()


#Lot of variables are not specifically normally distributed. Therefore transforming them using log(Base e) transform
train_data['Log_sqftlot'] = train_data['sqft_lot'].apply(lambda x: np.log(x))
train_data['Log_price'] = train_data['price'].apply(lambda x: np.log(x))
train_data['Log_sqftlot15'] = train_data['sqft_lot15'].apply(lambda x: np.log(x))
train_data['Log_sqftLiving'] = train_data['sqft_living'].apply(lambda x: np.log(x))

#GeoLocation to be added into the picture. Try to convert Latitude and Longitude to location names
for i in range(0, len(train_data['lat'])):
	coordiantes = (train_data.loc[i,'lat'], train_data.loc[i,'long'])
	train_data.loc[i, 'Location'] = rgc.search(coordiantes, mode = 1)[0].get('name')


mappingDictionary = {}

for k in train_data.Location.unique():
	#print(train_data[train_data['Location'] == k].price.mean())
	mappingDictionary[k] = train_data[train_data['Location'] == k].price.mean()

train_data['LocationMapping'] = train_data['Location'].apply(lambda x: mappingDictionary.get(x))

meanOfLocation = train_data['LocationMapping'].mean()
standardDeviationLocation = train_data['LocationMapping'].std()

train_data['LocationMapping'] = train_data['LocationMapping'].apply(lambda x: ((x-meanOfLocation)/standardDeviationLocation))

train_data['LivingSpaceAvailable'] = train_data['sqft_living']/train_data['sqft_lot']
train_data['NeighbourSpace'] = train_data['sqft_living15']/train_data['sqft_lot15']
train_data['Log_LivingSpaceAvailable'] = train_data['LivingSpaceAvailable'].apply(lambda x: np.log(x))
train_data['Log_NeighbourSpace'] = train_data['NeighbourSpace'].apply(lambda x: np.log(x))


#The Value of VIF tells that there is a collinearity between the Living space and above space. Assuming that both will be same if basement is not there.
#Therefore removing basement values and converting it into a variable to express the presence or absence of it
train_data['IsBasementThere'] = train_data['sqft_basement'].apply(lambda x: 1 if x >= 1 else -1)
#plt.hist(train_data['sqft_lot'], color = "red")
#plt.hist(train_data['BasementOccupancy'], color = "skyblue")
#plt.show()


#Check the correlation matrix with price (the below command only works with jupyter notebook or other software having HTML support)
train_data.corr().loc[:,['Log_price', 'price']].style.background_gradient(cmap='coolwarm', axis=None)


#Training Vector based on correlation and VIF and Chi Square Test
columnsToTrain = ['Log_sqftLiving','waterfront','floors','view', 'grade', 'HomeAgeinYear','LocationMapping', 'Log_LivingSpaceAvailable', 'Log_NeighbourSpace', 'RenovatedafterYears']

#Predicting the Log Price 
X, y = train_data.loc[:, columnsToTrain], train_data.loc[:, 'Log_price']

vif = pd.DataFrame()
New_X = add_constant(X)
vif['VIF Factors']  = [variance_inflation_factor(New_X.values, i) for i in range(New_X.shape[1])]
vif['Columns'] = New_X.columns
print(vif)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.30)

#Polynomial regression since columns like floors, view had polynomial relation with the log of price 
polynomialVariable = PolynomialFeatures(degree = 3)
polynomialCurveFitting = polynomialVariable.fit_transform(X_train)
polynomialCurveFittingTest = polynomialVariable.fit_transform(X_test)

Y_train_Exponential = [math.exp(x) for x in Y_train]
Y_test_Exponential = [math.exp(x) for x in Y_test]


#Ridge Regression
RidgeModel = Ridge(alpha = 50.0)
fittedRidgeModel = RidgeModel.fit(polynomialCurveFitting, Y_train)
scoreRidge = model_selection.cross_val_score(fittedRidgeModel, polynomialCurveFitting, Y_train, cv = 10) 

print('Ridge Regression Average Score', np.mean(scoreRidge))

PredictedRidgeDataTrain = fittedRidgeModel.predict(polynomialCurveFitting)
PredictedRidgeDataTrainExponential = [math.exp(x) for x in PredictedRidgeDataTrain]
PredictedRidgeData = fittedRidgeModel.predict(polynomialCurveFittingTest)
PredictedRidgeDataExponential = [math.exp(x) for x in PredictedRidgeData]

print('Root mean square Value Train Ridge Regression', np.sqrt(mean_squared_error(Y_train_Exponential, PredictedRidgeDataTrainExponential)))
print('Root mean square Value Test Ridge Regression', np.sqrt(mean_squared_error(Y_test_Exponential, PredictedRidgeDataExponential)))
#polynomialVariable.fit(X_train, Y_train)

#Support Vector Machine	Regressions
'''SupportVectorModel = LinearSVR()
fittedModelSVM = SupportVectorModel.fit(polynomialCurveFitting, Y_train)

scoresSVM = model_selection.cross_val_score(fittedModelSVM, polynomialCurveFitting, Y_train, cv = 10)
print(scoresSVM)


print('SupportVectorModel Score Train', SupportVectorModel.score(polynomialCurveFitting, Y_train))
print('SupportVectorModel Score Test', SupportVectorModel.score(polynomialCurveFittingTest, Y_test))


PredictSVMData = fittedModelSVM.predict(polynomialCurveFittingTest)
PredictSVMDataExponential = [math.exp(x) for x in PredictSVMData]

print('SVM Root mean square Error', np.sqrt(mean_squared_error(Y_test_Exponential, PredictSVMDataExponential)))
'''
#Fitting a linear model01
model = LinearRegression()
fittedModel = model.fit(X_train, Y_train)

#Fitting a polynomial model
model2 = LinearRegression()
fittedModel2 = model2.fit(polynomialCurveFitting, Y_train)

PredictedTrainData = fittedModel2.predict(polynomialCurveFitting)
PredictedTrainDataExponential = [math.exp(x) for x in PredictedTrainData]

print('RootMeanSquare Training', np.sqrt(mean_squared_error(Y_train_Exponential, PredictedTrainDataExponential)))

#Fitting the model on predicted set
PredictedTestData = fittedModel2.predict(polynomialCurveFittingTest)

PredictedTestDataExponential = [math.exp(x) for x in PredictedTestData]


print('RootMeanSquare Testing', np.sqrt(mean_squared_error(Y_test_Exponential, PredictedTestDataExponential)))

scores = model_selection.cross_val_score(model2, polynomialCurveFitting, Y_train, cv = 10)
scoresTest = model_selection.cross_val_score(model2, polynomialCurveFittingTest, Y_test, cv = 10)

print('Scores for Train and Test', np.mean(scores),scoresTest)
print(fittedModel.score(X_train, Y_train))
print(fittedModel.coef_)

print('Polynomial Regression Score', fittedModel2.score(polynomialCurveFitting, Y_train))

formuala = 'Log_price ~ Log_sqftLiving+waterfront+floors+view+grade+HomeAgeinYear+LocationMapping+Log_LivingSpaceAvailable+Log_NeighbourSpace+RenovatedafterYears'
statisticalModel = sm.ols(formuala, data = train_data)
statsfitted = statisticalModel.fit()
predictedStats = statsfitted.predict(X_test)
#print(predictedStats)
predictedStatsExponential = [math.exp(x) for x in predictedStats]
print('Root Mean Square OLS', np.sqrt(mean_squared_error(Y_test_Exponential, predictedStatsExponential)))
print(statsfitted.summary())

