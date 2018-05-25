import pandas as pd 
import numpy as np
from numpy import genfromtxt
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

import os
PATH=os.getenv('HOME')+'/.kaggle/competitions/home-credit-default-risk'
print(os.listdir(PATH))
x = pd.read_csv(PATH+"/application_train.csv")

print(np.shape(x.values))
y = x.pop('TARGET')
y.fillna(y.mean())
numerical_variables = list(x.dtypes [x.dtypes != "object"].index)
xNum = x[numerical_variables]
#x["AMT_REQ_CREDIT_BUREAU_DAY"].fillna(x["AMT_REQ_CREDIT_BUREAU_DAY"].mean())
#x["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(x["AMT_REQ_CREDIT_BUREAU_YEAR"].mean())
xNum.is_copy = False
for col in xNum:
	xNum[col].fillna(xNum[col].mean(),inplace=True)

model = RandomForestClassifier(n_estimators=100)
#model.fit(xNum, y)
cross_val_score(model, xNum, y, cv=10)

#x_train = application_train.values[:,[0])


#model.fit(x,y)
#xtrain, xtest, ytrain, ytest = (application_train)
#print(application_train['SK_ID_CURR'])


