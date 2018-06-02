import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import seaborn as sns

PATH = os.getenv('HOME')+'/.kaggle/competitions/home-credit-default-risk/'
#PATH = "/Users/Ojasvi/.kaggle/competitions/home-credit-default-risk/"

print("Importing data...")
dataframe = pd.read_csv(PATH + "application_train.csv")
test  = pd.read_csv(PATH + 'application_test.csv')
prev  = pd.read_csv(PATH + 'previous_application.csv')
bureau  = pd.read_csv(PATH + 'bureau.csv')
bureau_balance  = pd.read_csv(PATH + 'bureau_balance.csv')
credit_card  = pd.read_csv(PATH + 'credit_card_balance.csv')
POS_CASH  = pd.read_csv(PATH + 'POS_CASH_balance.csv')
payments  = pd.read_csv(PATH + 'installments_payments.csv')
lgbm_sub  = pd.read_csv(PATH + 'sample_submission.csv')

print('The shape of our features is:', dataframe.shape)

# Dropping rows with NaN values
# IMPUTATION HERE PLS
#dataframe.dropna(inplace=True)

# seperate target variable
labels = np.array(dataframe['TARGET'])
dataframe = dataframe.drop('TARGET', axis = 1)
feature_list = list(dataframe.columns)

# One Hot Encoding -  converts categorical data in training into numerical
cat_features = [col for col in dataframe.columns if dataframe[col].dtype == 'object']

# concatenate training and test and do one hot encoding
one_hot = pd.concat([dataframe, test])
one_hot = pd.get_dummies(one_hot, columns=cat_features)

# seperate back to training and test sets after one hot encoding
dataframe = one_hot.iloc[:dataframe.shape[0],:] 
test = one_hot.iloc[dataframe.shape[0]:,] 

print('The shape of our features is:', dataframe.shape)

#Pre-processing bureau_balance
print('Pre-processing bureau_balance...')
bureau_grouped_size = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
bureau_grouped_max = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
bureau_grouped_min = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

bureau_counts = bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
bureau_counts_unstacked = bureau_counts.unstack('STATUS')
bureau_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]
bureau_counts_unstacked['MONTHS_COUNT'] = bureau_grouped_size
bureau_counts_unstacked['MONTHS_MIN'] = bureau_grouped_min
bureau_counts_unstacked['MONTHS_MAX'] = bureau_grouped_max

bureau = bureau.join(bureau_counts_unstacked, how='left', on='SK_ID_BUREAU')

#Pre-processing previous_application
print('Pre-processing previous_application...')
#One-hot encoding of categorical features in previous application data set
prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
prev = pd.get_dummies(prev, columns=prev_cat_features)
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']

#Pre-processing bureau
print('Pre-processing bureau...')
#One-hot encoding of categorical features in bureau data set
bureau_cat_features = [bcol for bcol in bureau.columns if bureau[bcol].dtype == 'object']
bureau = pd.get_dummies(bureau, columns=bureau_cat_features)
avg_bureau = bureau.groupby('SK_ID_CURR').mean()
avg_bureau['bureau_count'] = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_bureau['SK_ID_BUREAU']

#Pre-processing POS_CASH
print('Pre-processing POS_CASH...')
le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Pre-processing credit_card
print('Pre-processing credit_card...')
credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Pre-processing payments
print('Pre-processing payments...')
avg_payments = payments.groupby('SK_ID_CURR').mean()
avg_payments2 = payments.groupby('SK_ID_CURR').max()
avg_payments3 = payments.groupby('SK_ID_CURR').min()
del avg_payments['SK_ID_PREV']

#Join data bases
print('Joining databases...')
dataframe= dataframe.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

dataframe= dataframe.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')

dataframe= dataframe.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

dataframe= dataframe.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

dataframe= dataframe.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')

dataframe= dataframe.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')

dataframe= dataframe.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')

#Remove features with many missing values
print('Removing features with more than 80% missing...')
test = test[test.columns[dataframe.isnull().mean() < 0.85]]
dataframe= dataframe[dataframe.columns[dataframe.isnull().mean() < 0.85]]

#Delete customer Id
del dataframe['SK_ID_CURR']
del test['SK_ID_CURR']

#Create train and validation set
train_x, valid_x, train_y, valid_y = train_test_split(dataframe, labels, test_size=0.2, shuffle=True)

#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x,label=train_y)
valid_data=lgb.Dataset(valid_x,label=valid_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'auc'
          }

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds= 40,
                 verbose_eval= 10
                 )

#Predict on test set and write to submit
predictions_lgbm_prob = lgbm.predict(test)

lgbm_sub.TARGET = predictions_lgbm_prob

lgbm_sub.to_csv('lgbm_submission.csv', index=False)

#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')
