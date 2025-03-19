# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: G.Hindhu
RegisterNumber: 212223230079 
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/e9ec5964-ec89-4ae8-b39d-ce6e7e82ac68)

```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![image](https://github.com/user-attachments/assets/fb0e32b1-9dfd-449f-96f7-9ef12ba1a3e7)

```
data1.isnull()
```
![image](https://github.com/user-attachments/assets/2e19bb74-9022-4e05-aabc-868776de6f02)

```
data1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/f570cba1-6b1d-4eff-b610-5878c7d8d1af)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1
```
![image](https://github.com/user-attachments/assets/11e610df-ae2a-4e7e-9fb7-fcfacf8f8c33)

```
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/a9d4cd76-aec6-4073-99c1-98971a411735)

```
y=data1["status"]
y
```
![image](https://github.com/user-attachments/assets/a4506fd9-f11e-44e6-b6f0-4af65067b868)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/18dc68e0-0127-49e1-a743-263209f589a8)

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/de7869f7-f75f-4c9f-8c27-2237d0a8716a)

```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/fa79b83c-724c-4cfe-8172-8af3da5d88eb)

```
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/5600508e-fb34-4c5f-9443-5aad14b9c827)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
