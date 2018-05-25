import pandas as pd 
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import os
PATH=os.getenv('HOME')+'/.kaggle/competitions/home-credit-default-risk'
print(os.listdir(PATH))
application_train = pd.read_csv(PATH+"/application_train.csv")

print(np.shape(application_train.values))
y = application_train.pop('TARGET')
y.fillna(y.mean())
x = application_train
numerical_variables = list(x.dtypes [x.dtypes != "object"].index)
x = x[numerical_variables]
#x["AMT_REQ_CREDIT_BUREAU_DAY"].fillna(x["AMT_REQ_CREDIT_BUREAU_DAY"].mean())
#x["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(x["AMT_REQ_CREDIT_BUREAU_YEAR"].mean())
x.is_copy = False
for col in x:
	x[col].fillna(x[col].mean(),inplace=True)

model = RandomForestClassifier(n_estimators=100)
#model.fit(x, y)


#x_train = application_train.values[:,[0])


#model.fit(x,y)
#xtrain, xtest, ytrain, ytest = (application_train)
#print(application_train['SK_ID_CURR'])


