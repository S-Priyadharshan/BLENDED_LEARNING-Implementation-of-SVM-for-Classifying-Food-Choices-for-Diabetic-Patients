# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Data Preparation: Collect and preprocess data on food attributes and diabetic suitability labels, scaling features for SVM compatibility.
Model Training: Split the data into training and test sets, then train an SVM model using the training set with a suitable kernel.
Model Evaluation: Evaluate the SVM on the test set using metrics like accuracy, precision, and recall to measure performance.
Prediction: Use the trained SVM model to classify new food choices as suitable or unsuitable for diabetic patients.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: Priyadharshan S
RegisterNumber: 212223240127  
*/

# Import required packages
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
# Evaluation metrics related methods
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
%matplotlib inline
# Setup a random seed to be 123
rs = 123
# Load the dataset
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/food_items_binary.csv"
food_df = pd.read_csv(dataset_url)
# Get the row entries with col 0 to -1 (16)
feature_cols = list(food_df.iloc[:, :-1].columns)
feature_cols
X = food_df.iloc[:, :-1]
y = food_df.iloc[:, -1:]
# # Get the row entries with the last col 'class'
y.value_counts(normalize=True)
y.value_counts().plot.bar(color=['red', 'green'])
# First, let's split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)
model = SVC()
model.fit(X_train, y_train.values.ravel())
preds = model.predict(X_test)
def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos
evaluate_metrics(y_test, preds)
model = SVC(C=10, kernel='rbf')
model.fit(X_train, y_train.values.ravel())
preds = model.predict(X_test)
evaluate_metrics(y_test, preds)
```

## Output:
![image](https://github.com/user-attachments/assets/dff32c33-0c77-4ca9-a584-08ac0217a5cf)
![image](https://github.com/user-attachments/assets/fe83eea1-d577-470c-8e51-177f810beb40)
![image](https://github.com/user-attachments/assets/66ac29a2-b529-495e-9f26-6ea1ab087d11)


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
