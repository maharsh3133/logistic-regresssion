#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:41:01 2023

@author: maharsh
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

titaninc_maharsh=pd.read_csv(r'titanic.csv')

print(titaninc_maharsh.head(3))
print("_________________________________________________________________\n")
print(titaninc_maharsh.shape)
print("_________________________________________________________________\n")
print(titaninc_maharsh.columns.values)
print("_________________________________________________________________\n")
print(titaninc_maharsh.info())
print("_________________________________________________________________\n")
print("Total null values per columns:")
print(titaninc_maharsh.isnull().sum())
print("_________________________________________________________________\n")
print("Number of Unique values in colums:")
print("Name : "+str(len(titaninc_maharsh['Name'].unique())))
print("Ticket : "+str(len(titaninc_maharsh['Ticket'].unique())))
print("Cabin : "+str(len(titaninc_maharsh['Cabin'].unique())))
print("_________________________________________________________________\n")
print("Unique values in colums:")
print("Embarked : "+str(titaninc_maharsh['Embarked'].unique()))
print("Sex : "+str(titaninc_maharsh['Sex'].unique()))
print("Pclass : "+str(titaninc_maharsh['Pclass'].unique()))
print("_________________________________________________________________\n")
print('dropping unwanted columns : "Name","PassengerId","Cabin","Ticket" ')
titaninc_maharsh=titaninc_maharsh.drop(columns=["Name","PassengerId","Cabin","Ticket"])

print("_________________________________________________________________\n")
print("plotting BAR graphs")
pd.crosstab(titaninc_maharsh["Pclass"],titaninc_maharsh["Survived"]).plot(kind='bar')
pd.crosstab(titaninc_maharsh["Sex"],titaninc_maharsh["Survived"]).plot(kind='bar')
print("_________________________________________________________________\n")
print('Converting categorical to nemeric :"Sex","Embarked" ')

titaninc_maharsh=pd.get_dummies(titaninc_maharsh,columns=["Sex","Embarked"],drop_first=False)
print(titaninc_maharsh.info())

titaninc_maharsh=titaninc_maharsh.astype(np.float32)
print(titaninc_maharsh.info())

print("_________________________________________________________________\n")
print("plot scatter matrix")
pd.plotting.scatter_matrix(titaninc_maharsh, alpha=.4,figsize=(20,25))
plt.show()

print("_________________________________________________________________\n")
print("filling null values in age column with the mean of the column")
titaninc_maharsh['Age'].fillna(int(titaninc_maharsh['Age'].mean()), inplace=True)
print("_________________________________________________________________\n")
print("convert all data type to float")
titaninc_maharsh=titaninc_maharsh.astype(float)
print(titaninc_maharsh.info())

print("_________________________________________________________________\n")
print("normalize the data")
def normalize(data):
    return (data-data.min())/(data.max()-data.min())


normal_data=normalize(titaninc_maharsh)

print(normal_data.head())
print("_________________________________________________________________\n")
print("plot histogram")
hist=normal_data.hist(figsize=(9,10))

print("_________________________________________________________________\n")
print("Feature and target Selection")
data_mayy_b_final_vars=normal_data.columns.values.tolist()
Y=['Survived']
X=[i for i in data_mayy_b_final_vars if i not in Y ]

X=normal_data[X]
Y=normal_data[Y]
print(X,Y)

print("_________________________________________________________________\n")
print("Evaluating the model by runnign evaluation from 10% to 50% of test data")
for i in np.arange(0.10,0.5,0.05):
    print("\n\ntest for " + str(i*100)+"% of data")
    x_train_maharsh, x_test_maharsh, y_train_maharsh, y_test_maharsh = train_test_split(X, np.ravel(Y), test_size=i,random_state=2)
    maharsh_model = linear_model.LogisticRegression(solver='lbfgs')
    maharsh_model.fit(x_train_maharsh, y_train_maharsh)
    scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), x_train_maharsh, y_train_maharsh, scoring='accuracy', cv=10)
    print (scores.ravel())
    print ("Mean score: "+str(scores.mean()))
    print ("Max score: "+str(scores.max()))
    print ("Min score: "+str(scores.min()))

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


print("_________________________________________________________________\n")
print("train test split 70/30")
x_train_maharsh, x_test_maharsh, y_train_maharsh, y_test_maharsh = train_test_split(X, Y, test_size=0.30,random_state=2)

print(x_train_maharsh, x_test_maharsh, y_train_maharsh, y_test_maharsh)


print("_________________________________________________________________\n")
print("defining and training model on training data")
maharsh_model = linear_model.LogisticRegression(solver='lbfgs')
maharsh_model.fit(x_train_maharsh, np.ravel(y_train_maharsh))

result= pd.DataFrame(zip(x_train_maharsh.columns, np.transpose(maharsh_model.coef_.ravel())))
print(result)


print("_________________________________________________________________\n")
print("Score")

scores = cross_val_score(linear_model.LogisticRegression(solver='lbfgs'), x_train_maharsh, np.ravel(y_train_maharsh), scoring='accuracy', cv=10)
print (scores.ravel())
print (scores.mean())

print("_________________________________________________________________\n")
print("prediction with thersold 0.5")

y_pred_maharsh_train = maharsh_model.predict_proba(x_train_maharsh)
#print(y_pred_maharsh)
type(y_pred_maharsh_train)

y_pred_maharsh_train_flag = (y_pred_maharsh_train[:,1]>0.5)
#print (y_pred_maharsh_flag)

print("_________________________________________________________________\n")
print("Accuracy of the model on train data")
print (metrics.accuracy_score(y_train_maharsh, y_pred_maharsh_train_flag))


y_pred_maharsh = maharsh_model.predict_proba(x_test_maharsh)
#print(y_pred_maharsh)
type(y_pred_maharsh)

y_pred_maharsh_flag = (y_pred_maharsh[:,1]>0.5)
#print (y_pred_maharsh_flag)

print("_________________________________________________________________\n")
print("Accuracy of the model on test data")
print (metrics.accuracy_score(y_test_maharsh, y_pred_maharsh_flag))

print("_________________________________________________________________\n")
print("Accuracy of the model for 0.5 threshold")
print (metrics.accuracy_score(y_test_maharsh, y_pred_maharsh_flag))

print("_________________________________________________________________\n")
print("Confusion metrix 0.5 threshold")
Y_test=(y_test_maharsh==1)
Y_A=Y_test.values
confusionmatrix = confusion_matrix(Y_A, y_pred_maharsh_flag)
print(confusionmatrix)

print("_________________________________________________________________\n")
print("Cclassifiction report for 0.5 thresoled")
print(classification_report(Y_A, y_pred_maharsh_flag))



print("_________________________________________________________________\n")
print("prediction with thersold 0.75")

y_pred_maharsh_flag = (y_pred_maharsh[:,1]>0.75)
#print (y_pred_maharsh_flag)
print("_________________________________________________________________\n")
print("Accuracy of the model for 0.75 threshold")
print (metrics.accuracy_score(y_test_maharsh, y_pred_maharsh_flag))

print("_________________________________________________________________\n")
print("Confusion metrix 0.75 threshold")
Y_test=(y_test_maharsh==1)
Y_A=Y_test.values
confusion_matrix = confusion_matrix(Y_A, y_pred_maharsh_flag)
print(confusion_matrix)

print("_________________________________________________________________\n")
print("Cclassifiction report for 0.75 thresoled")
print(classification_report(Y_A, y_pred_maharsh_flag))

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")















