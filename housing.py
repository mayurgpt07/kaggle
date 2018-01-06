from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score

train_data = pd.read_csv("./data/train_housing.csv")
test_data = pd.read_csv("./data/test_housing.csv")

print(train_data.corr().iloc[:,37])
# print(train_data['Fence'])
train_data['Fence'] = train_data['Fence'].replace(np.NaN, 'NA')
train_data['PoolQC'] = train_data['PoolQC'].replace(np.NaN, 'NA')
train_data['GarageCond'] = train_data['GarageCond'].replace(np.NaN, 'NA')
train_data['GarageQual'] = train_data['GarageQual'].replace(np.NaN, 'NA')
train_data['GarageFinish'] = train_data['GarageFinish'].replace(np.NaN, 'NA')
train_data['FireplaceQu'] = train_data['FireplaceQu'].replace(np.NaN, 'NA')
train_data['Electrical'] = train_data['Electrical'].replace(np.NaN, 'NA')
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].replace(np.NaN, 'NA')
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].replace(np.NaN, 'NA')
train_data['BsmtExposure'] = train_data['BsmtExposure'].replace(np.NaN, 'NA')
train_data['BsmtCond'] = train_data['BsmtCond'].replace(np.NaN, 'NA')
train_data['BsmtQual'] = train_data['BsmtQual'].replace(np.NaN, 'NA')

# print(train_data['Fence'])
dict_list = {'LotShape': ['NA','IR3','IR2','IR1','Reg'],
             'Utilities':['NA','ELO','NoSeWa','NoSewr','AllPub'],
             'LandSlope':['NA','Sev','Mod','Gtl'],
             'ExterQual':['NA','Po','Fa','TA','Gd','Ex'],
             'ExterCond':['NA','Po','Fa','TA','Gd','Ex'],
             'BsmtQual':['NA','Po','Fa','TA','Gd','Ex'],
             'BsmtCond':['NA','Po','Fa','TA','Gd','Ex'],
             'BsmtExposure':['NA','No','Mn','Av','Gd'],#check
             'BsmtFinType1':['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],#check
             'BsmtFinType2':['NA','Unf','LwQ','Rec','BLQ','ALQ','GLQ'],#check
             'HeatingQC':['NA','Po','Fa','TA','Gd','Ex'],
             'Electrical':['NA','Mix','FuseP','FuseF','FuseA','SBrkr'],#check
             'KitchenQual':['NA','Po','Fa','TA','Gd','Ex'],
             'Functional':['NA','Sal','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],
             'FireplaceQu':['NA','Po','Fa','TA','Gd','Ex'],#check
             'GarageFinish':['NA','Unf','RFn','Fin'],#check
             'GarageQual':['NA','Po','Fa','TA','Gd','Ex'],#check
             'GarageCond':['NA','Po','Fa','TA','Gd','Ex'], #check
             'PavedDrive':['NA','N','P','Y'],
             'PoolQC':['NA','Po','Fa','TA','Gd','Ex'], #check
             'Fence':['NA','MnWw','GdWo','MnPrv','GdPrv']} #check
dict_list_values = {'LotShape': [],
                    'Utilities': [],
                    'LandSlope': [],
                    'ExterQual': [],
                    'ExterCond': [],
                    'BsmtQual': [],
                    'BsmtCond': [],
                    'BsmtExposure': [],
                    'BsmtFinType1': [],
                    'BsmtFinType2': [],
                    'HeatingQC': [],
                    'Electrical': [],
                    'KitchenQual': [],
                    'Functional': [],
                    'FireplaceQu': [],
                    'GarageFinish': [],
                    'GarageQual': [],
                    'GarageCond': [],
                    'PavedDrive': [],
                    'PoolQC': [],
                    'Fence': []}

for key in dict_list:
    # print(key)
    for i in train_data[key]:
        if i in dict_list[key]:
            dict_list_values[key].append(dict_list[key].index(i))
        else:
            print(i)
            print(key)
    # print(len(dict_list_values[key]))
column_header = []
for key in dict_list:
    column_header.append(key)

df = pd.DataFrame(columns=column_header)
df = df.from_dict(dict_list_values)
print(df)
# values = np.where(train_data['MSSubClass'] == 20,pd.to_numeric(train_data['SalePrice'],downcast='integer'), np.NaN)
# print(values)
# plt.bar(train_data['MSSubClass'].unique(), train_data['SalePrice'])
# plt.show()
