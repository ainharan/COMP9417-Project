import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("C:/Users/Ojasvi/.kaggle/competitions/home-credit-default-risk/application_train.csv")

print('The shape of our features is:', dataframe.shape)

# Dropping rows with NaN values
# IMPUTATION HERE PLS
dataframe.dropna(inplace=True)

print('The shape of our features is:', dataframe.shape)

# One Hot Encoding
dataframe = pd.get_dummies(dataframe)

print('The shape of our features is:', dataframe.shape)

# Only using 500 rows because training takes so long
dataframe = dataframe.head(500)

labels = np.array(dataframe['TARGET'])

dataframe = dataframe.drop('TARGET', axis = 1)
feature_list = list(dataframe.columns)

dataframe = np.array(dataframe)


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(dataframe, labels, test_size = 0.20, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 50, max_depth=200, max_features=200, random_state = 42)
rf2 = RandomForestRegressor(n_estimators = 1000, max_depth=200, max_features=200, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

#print(rf.feature_importances_)
important_features = pd.Series(data=rf.feature_importances_,index=feature_list)
important_features.sort_values(ascending=False,inplace=True)

#print(important_features)

#s = pd.Series(important_features)

#s.plot.bar()

#plt.show()

# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, train_features, train_labels, cv=kfold, scoring='accuracy')
print(result.mean())
