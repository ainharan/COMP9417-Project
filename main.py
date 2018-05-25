import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
# matplotlib inline 
import os
PATH=os.getenv('HOME')+'/.kaggle/competitions/home-credit-default-risk'
print(os.listdir(PATH))
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")