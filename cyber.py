import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load the dataset into a Pandas DataFrame
df = pd.read_csv('station_data_dataverse.csv')

# Print all of the features in the dataset
print(df.columns)
# read in dataset
df = pd.read_csv('station_data_dataverse.csv')

# add FalseDataInjection column with random 0 or 1 values
df['FalseDataInjection'] = np.random.randint(2, size=len(df))

# save new dataset to CSV file
df.to_csv('dataset_with_FDI.csv', index=False)
df.head()
df
# Check for any null or missing values in the DataFrame
print(df.isnull().sum())
# Drop all rows with null or empty values
df = df.dropna()

# Save the cleaned dataset to a new csv file
df.to_csv('cleaned_dataset.csv', index=False)
# read cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

# print all features
print(df.columns)
# Perform one-hot encoding on the 'Weekday' feature
one_hot = pd.get_dummies(df['weekday'], prefix='weekday')

# Add the new one-hot encoded features to the original dataset
df = pd.concat([df, one_hot], axis=1)

# Drop the original 'Weekday' feature since it's no longer needed
df.drop('weekday', axis=1, inplace=True)

# Print the updated dataset
print(df.head())
df
# Select the features for SVM
features = ['kwhTotal', 'dollars', 'chargeTimeHrs', 'distance', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'managerVehicle']

# Split the data into X (features) and y (target)
X = df[features]
y = df['FalseDataInjection']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM classifier with RBF kernel
clf = svm.SVC(kernel='rbf')

# Train the SVM classifier on the training data
clf.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = clf.predict(X_test)
print(y_pred)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % accuracy)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g')
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
import os
print(os.getcwd())
