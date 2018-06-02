import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
from DataFrameImputer import DataFrameImputer

PATH = os.getenv('HOME')+'/.kaggle/competitions/home-credit-default-risk/'
#PATH = "/Users/Ojasvi/.kaggle/competitions/home-credit-default-risk/"

print("Importing data...")
dataframe = pd.read_csv(PATH + "application_train.csv")
test  = pd.read_csv(PATH + 'application_test.csv')
#prev  = pd.read_csv(PATH + 'previous_application.csv')
#bureau  = pd.read_csv(PATH + 'bureau.csv')
#bureau_balance  = pd.read_csv(PATH + 'bureau_balance.csv')
#credit_card  = pd.read_csv(PATH + 'credit_card_balance.csv')
#POS_CASH  = pd.read_csv(PATH + 'POS_CASH_balance.csv')
#payments  = pd.read_csv(PATH + 'installments_payments.csv')
#lgbm_sub  = pd.read_csv(PATH + 'sample_submission.csv')

# seperate target variable
labels = np.array(dataframe['TARGET'])
dataframe = dataframe.drop('TARGET', axis = 1)
feature_list = list(dataframe.columns)

#Remove features with many missing values
print('Removing features with more than 80% missing...')
test = test[test.columns[dataframe.isnull().mean() < 0.85]]
dataframe= dataframe[dataframe.columns[dataframe.isnull().mean() < 0.85]]

print("Imputing Data...")
# imputes with the mean of each col
#for col in dataframe:
#	dataframe[col].fillna(dataframe[col].mean())
#dataframe = dataframe.fillna(dataframe.mean())
#dataframe.dropna(inplace=True)
X = pd.DataFrame(dataframe)
dataframe = DataFrameImputer().fit_transform(X)

# get numeric variables
numeric_variables = list(dataframe.dtypes[dataframe.dtypes != "object"].index)

print("One hot encoding...")
# One Hot Encoding -  converts categorical data in training into numerical
cat_features = [col for col in dataframe.columns if dataframe[col].dtype == 'object']

# concatenate training and test and do one hot encoding
one_hot = pd.concat([dataframe, test])
one_hot = pd.get_dummies(one_hot, columns=cat_features)

# seperate back to training and test sets after one hot encoding
dataframe = one_hot.iloc[:dataframe.shape[0],:] 
test = one_hot.iloc[dataframe.shape[0]:,] 


print("fitting baseline model on just numerical values...")
# fit model on just numerical variables as a baseline
model = RandomForestRegressor(n_estimators=2, oob_score=True, random_state=42)
model.fit(dataframe[numeric_variables], labels)

# for regression the oob_score_ (out of bag score) gives the R^2 based on oob predictions
print(model.oob_score_)

labels_oob = model.oob_predictions_
print("c-stat ", roc_auc_score(labels, labels_oob))
print("Out of bag score...")
print(labels_oob)
print("Printing importance matrix")
print(model.feature_importances_)

feature_importances = pd.Series(model.feature_importances_, index=dataframe.columns)
feature_importances.sort()
feature_importances.plot(kind="barh")
plt.show()
