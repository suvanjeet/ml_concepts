# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:56:35 2018

@author: suvanjeet
"""

#%%

import pandas as pd
import numpy as np

pd.set_option('display.max_columns',None)    # print all columns
pd.set_option('display.max_rows',None)       # print all rows 
#adult=pd.read_csv(r'D:\Python day executions\files\adult_data.csv',header=None) 
train_data=pd.read_csv(r'D:\Python day executions\files\riskanalytics\risk_analytics_train.csv',header=0) 
test_data=pd.read_csv(r'D:\Python day executions\files\riskanalytics\risk_analytics_test.csv',header=0) 

print(train_data.head())
print(test_data.head())
#%%

print(train_data.isnull().any(axis=1).sum())

#%%
print(train_data.shape)
print(test_data.shape)

#%%
print(train_data.info())

#%%
print(train_data.isnull().sum())

#%%

print(train_data.describe(include='all'))

'''
Gender,Married,Dependents,Self_Employed,Loan_Amount_Term --mode

Credit_History is int, but if find mean then will be risk , so initially make all those zero.

Dependents,Loan_Amount_Term,Credit_History are discread int value so we will go with mode value

LoanAmount  -- mean '''

#%%

colname1=['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term']

for x in colname1:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)

print(train_data.isnull().sum())


#%%

train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(),inplace=True)
print(train_data.isnull().sum())

#%%

#importing values for credit history

train_data['Credit_History'].fillna(value=0,inplace=True)
print(train_data.isnull().sum())

#%%
# transforming categorical to numericalvalue

from sklearn import preprocessing

colname=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
le={}

for x in colname:
    le[x]= preprocessing.LabelEncoder()
for x in colname:
    train_data[x]= le[x].fit_transform(train_data[x])
    
#%%
    
    print(train_data.head())


#%%
    
x_train=train_data.values[:,1:-1]
y_train=train_data.values[:,-1]

#%%
y_train=y_train.astype(int)


#%%



#%%
test_data.info()
print(test_data.isnull().sum())


#%%

colname1=['Gender','Dependents','Self_Employed','Loan_Amount_Term']

for x in colname1:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)

print(test_data.isnull().sum())


#%%

test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean(),inplace=True)
print(test_data.isnull().sum())

#%%

#importing values for credit history

test_data['Credit_History'].fillna(value=0,inplace=True)
print(test_data.isnull().sum())
#%%

test_data.info()

#%%
# transforming categorical to numericalvalue

from sklearn import preprocessing

colname=['Gender','Married','Education','Self_Employed','Property_Area']
le={}

for x in colname:
    le[x]= preprocessing.LabelEncoder()
for x in colname:
    test_data[x]= le[x].fit_transform(test_data[x])

#%%
print(test_data.isnull().sum())

#%%
x_test=test_data.values[:,1:]

#%%

print(test_data.head())

#%%
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


#%%

from sklearn import svm

svc_model= svm.SVC(kernel='rbf',C=5.0,gamma=0.75)

'''from sklearn.linear_model import LogisticRegression
svc_model= LogisticRegression()'''
svc_model.fit(x_train,y_train)

y_pred=svc_model.predict(x_test)
print(list(y_pred))

#%%
y_pred1=svc_model.predict(x_train)
print(list(zip(y_pred1,y_train)))

#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_train,y_pred1)
print(cfm)                              # to print confusion matrix , classification details

print("classification report")

print(classification_report(y_train,y_pred1))  #recall - howmuch % +ve tc did we catch

acc= accuracy_score(y_train,y_pred1)
print("accuracy of the model: ", acc)   # to print the accuracy


#%%
y_pred_col=list(y_pred)

test_data=pd.read_csv(r'D:\Python day executions\files\riskanalytics\risk_analytics_test.csv',header=0) 
test_data['y_predictions']=y_pred_col
test_data.head()

#%%
test_data.to_csv('test_data.csv')


#%%
from sklearn import cross_validation
#performing kfold_cross_validation
kfold=cross_validation.KFold(n=len(x_train),n_folds=10)
print(kfold)

#%%

#%%
# cross validation process

classifier=svm.SVC(kernel='rbf', C=10, gamma=0. c  1)
#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,y=y_train, cv=kfold)
print(kfold_cv_result)
#%%
#finding the mean
print(kfold_cv_result.mean())

#%%
from sklearn import datasets
import pandas as pd
import numpy as np
 
iris=datasets.load_iris()

#%%

#iris.data
#iris.feature_names
#iris.target
iris.DESCR

#%%

x=iris.data[:,:]
y=iris.target[:]

#%%
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

scaler.fit(x)
x=scaler.transform(x)
y=y.astype(int)

#%%
#splitting the data in test and train
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=10)

#%%

from sklearn import svm

svc_model= svm.SVC(kernel='rbf',C=1.0,gamma=0.1)

'''from sklearn.linear_model import LogisticRegression
svc_model= LogisticRegression()'''
svc_model.fit(x_train,y_train)
y_pred=svc_model.predict(x_test)
print(list(zip(y_test,y_pred))

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details

print("classification report")

print(classification_report(y_test,y_pred))  #recall - howmuch % +ve tc did we catch

acc= accuracy_score(y_test,y_pred)
print("accuracy of the model: ", acc)   # to print the accuracy





