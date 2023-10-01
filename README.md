# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: YUVARAJ B
RegisterNumber: 212222230182

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
  
```

## Output:

# Placement data:

![270335971-d0ccab75-256f-4f14-b1e1-3f93cc0ae92a](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/20367bc7-948e-4642-a628-85d8ed9abb4c)

# Salary data:

![270335995-d9982d36-38e7-44dc-924b-15f8d96a5fff](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/7f649ec5-d6d0-4b18-96ef-4731d22ecf72)

# checking the null()functiom:

![270336078-75b24d78-a5f7-4269-8214-274cc416a0de](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/0f6a5d2e-60fb-4ea2-93e8-a4e4fd7ce5c6)

# Data duplicate:

![270336101-d89cf75f-e006-4e92-b827-9e7442f8523e](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/99528627-2252-4f55-874a-062f8773a1cd)

# print data:

![270336227-f2ff24f1-99d4-41d0-8687-ea720bd1f0aa](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/597caa02-0f61-4881-be0b-392d147b36ab)

# Data status:

![270336258-ba928a71-4c5f-4350-b12d-80e05ba607ff](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/878afc71-f14b-479f-9be3-d9aa7613980c)

# y_prediction array:

![270336272-eec92ff3-0571-405e-ae6e-6c2355348f5b](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/fcf2e446-f8a0-4c2c-a97c-653326d9adad)

# Accuracy value:

![270336290-2a0e0ffd-588a-4c76-a63f-53f4013fb05e](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/957e295d-2b03-4f06-8a6c-bd04ed42b019)

# confusion array:

![270336298-c6d6239f-991f-4042-8a53-7b733d681466](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/19ba7ec2-ed0a-4fee-834c-78964e3d530d)

# classification report:

![270336426-9dc02e02-90d2-4bd1-b8d9-7a3f5beee06a](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/276ebd5e-34f0-4c44-9380-db8a9592c6db)

# Prediction of LR:

![270336455-01702721-a55a-43ad-b76f-8a85d22902e6](https://github.com/Yuva2005raj/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118343998/733e7ee4-bc78-4069-a396-e81c294c9f2c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
