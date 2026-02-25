# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Bring in the necessary libraries.
2.Load the Dataset: Load the dataset into your environment.
3.Data Preprocessing: Handle any missing data and encode categorical variables as needed.
4.Define Features and Target: Split the dataset into features (X) and the target variable (y).
5.Split Data: Divide the dataset into training and testing sets.
6.Build Multiple Linear Regression Model: Initialize and create a multiple linear regression model.
7.Train the Model: Fit the model to the training data.
8.Evaluate Performance: Assess the model's performance using cross-validation.
9.Display Model Parameters: Output the model’s coefficients and intercept.
10.Make Predictions & Compare: Predict outcomes and compare them to the actual values.

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by:gowtham u 
RegisterNumber:212225040099
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')
data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)
print(data.head())
x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
print('Name: Lokeshwarn G')
print('Reg No:212225040210')
print("\n===Cross-Validation ===")
cv_scores = cross_val_score(model,x,y,cv=5)
print("Fold R^2 scores:",[f"{score:.2f}" for score in cv_scores])
print(f"Average R^2:{cv_scores.mean():.4f}")
y_pred = model.predict(x_test)
print("\n===Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()  
*/
```

## Output:
~~~
Screenshot 2026-02-26 005141.png
Screenshot 2026-02-26 005155.png
Screenshot 2026-02-26 005209.png
~~~


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
