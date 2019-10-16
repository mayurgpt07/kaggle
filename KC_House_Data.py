from sklearn import linear_model
from sklearn import model_selection
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


train_data = pd.read_csv("./data/kc_house_data.csv", sep = ",", header = 0)

#Start PreProcessing 
train_data['dateInFormat'] = train_data['date'].apply(lambda x: x[0:8])
train_data['dateInFormat'] = train_data['dateInFormat'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))

print(train_data)
#Check the correlation matrix with price
train_data.corr().loc[:,'id':'price'].style.background_gradient(cmap='coolwarm', axis=None)

pd.plotting.scatter_matrix(train_data.loc[:,'price':'floors'], alpha = 0.2, figsize = (20, 20), diagonal = 'kde')
plt.show()

X, y = train_data.loc[:, 'bedrooms': 'dateInFormat'], train_data.loc[:, 'price']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.33)


