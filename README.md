# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and preprocess data – Gather nutrition info, handle missing values, and encode features.
2.Split data – Divide into training and test sets.
3.Train model – Fit a Logistic Regression model on the training data.
4.Evaluate predictions – Test the model on test data and check accuracy.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('food_items_binary.csv')

print('Dataset Overview')
print(df.head())

print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:, :-1]
y_raw = df.iloc[:, -1:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw.values.ravel())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

penalty = 'l2'
multi_class = 'multinomial'
solver = 'lbfgs'
max_iter = 1000

l2_model = LogisticRegression(random_state=123, penalty=penalty, multi_class=multi_class, solver=solver, max_iter=max_iter)

l2_model.fit(X_train, y_train)

y_pred = l2_model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("Name: Lenasri R")
print("Reg. No: 212225040199")
```

## Output:
<img width="866" height="638" alt="image" src="https://github.com/user-attachments/assets/a37a0c9e-9157-404c-a8e4-138db053cdd8" />
<img width="615" height="687" alt="image" src="https://github.com/user-attachments/assets/fb0f2d3a-673b-44cf-9ce3-754944273e36" />
<img width="657" height="381" alt="image" src="https://github.com/user-attachments/assets/fd47f2f2-025b-4f60-8e17-3c26ad390124" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
