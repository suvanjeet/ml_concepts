# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 15:53:33 2018

@author: suvanjeet
"""

#%%
import pandas as pd
import numpy as np

#%%
pd.set_option('display.max_columns',None)

#%%
network_data=pd.read_csv(r'D:\Python day executions\files\networkintrusion\OBS_Network_data.csv',header=None, delimiter=' *, *', engine='python') 

network_data.head()
#%%
network_data.info()

#%%
network_data.shape

#%%
network_data.columns=["Node","Utilised Bandwith Rate","Packet Drop Rate","Full_Bandwidth","Average_Delay_Time_Per_Sec",
"Percentage_Of_Lost_Pcaket_Rate","Percentage_Of_Lost_Byte_Rate","Packet Received Rate","of Used_Bandwidth",
"Lost_Bandwidth","Packet Size_Byte","Packet_Transmitted","Packet_Received","Packet_lost","Transmitted_Byte",
"Received_Byte","10-Run-AVG-Drop-Rate","10-Run-AVG-Bandwith-Use","10-Run-Delay","Node Status","Flood Status","Class"]

#%%
network_data.head()

#%%
network_data.isnull().sum()
#%%

#create a copy of data frame

network_data_rev=pd.DataFrame.copy(network_data)
network_data_rev.head()

#%%
#to drop the variable
network_data_rev=network_data_rev.drop("Packet Size_Byte", axis=1)
network_data_rev.shape
#%%
colname=['Node','Full_Bandwidth','Node Status','Class']
colname

#%%
 # for preprocessing the data

from sklearn import preprocessing

le={}     # dictinary as it will be in key and value pair


for x in colname:
    le[x]=preprocessing.LabelEncoder()   # will lebel all categorical var in columns
    
for x in colname:
    network_data_rev[x]=le[x].fit_transform(network_data_rev[x])    # will replace the label with the variable data labeled
network_data_rev.describe(include='all')

#%%

network_data_rev.dtypes

#%%
network_data_rev.head()
#%%
x= network_data_rev.values[:,:-1]    # includes the columns other than income
y= network_data_rev.values[:,-1] 
print(y)

#%%
from sklearn.preprocessing import  StandardScaler
scaler= StandardScaler()
scaler.fit(x)       # only for train data

x= scaler.transform(x)  # for test and train both
print(x)

y=y.astype(int)


#%%
#splitting the data in test and train
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=10)

#%%
from sklearn.tree import DecisionTreeClassifier

model_decisiontree= DecisionTreeClassifier()
model_decisiontree.fit(x_train,y_train)
#%%
y_pred=model_decisiontree.predict(x_test)
print(list(zip(y_test,y_pred)))

#%%
model_decisiontree.score(x_test, y_test)


#%%
# to see the tree graphically

from sklearn import tree
with open(r"D:\Python day executions\files\model_Decisiontree.txt","w") as f:           # w - write mode, save the file in directory
    f=tree.export_graphviz(model_decisiontree.txt,out_file=f)       # file=f   store the steps

'''after this step text file will be generated at the specified path.   open the text file and copy the contains
now open http://webgraphviz.com/ in browser and paste the text details and click on generate graph'''

#%%


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 

#%%
#Using cross validation

model_decisiontree1=(DecisionTreeClassifier())

from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(x_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=model_decisiontree1,X=x_train,y=y_train, cv=kfold_cv)
print(kfold_cv_result)
#%%
#finding the mean
print(kfold_cv_result.mean())

#print(list(zip(Y_test,Y_pred)))


#%%

# using svm
from sklearn import svm

svc_model= svm.SVC(kernel='rbf',C=15.0,gamma=0.6)

'''from sklearn.linear_model import LogisticRegression
svc_model= LogisticRegression()'''
svc_model.fit(x_train,y_train)

y_pred=svc_model.predict(x_test)
#print(list(zip(y_test,y_pred)))

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 

#%%
#using logistic
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data in the model
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
y_pred

#%%
cfm1= confusion_matrix(y_test,y_pred)
print(cfm1)
acc= accuracy_score(y_test,y_pred)
print("accuracy of the model: ", acc)

print(classification_report(y_test,y_pred))

#%%

#Ensemble modelling
#predicting using the ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier

model=(ExtraTreesClassifier(21))

#fit the model on the data and predict the values

model=model.fit(x_train,y_train)

y_pred= model.predict(x_test)
#%%

?ExtraTreesClassifier

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 

#%%
#predicting using the RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

RFmodel=(RandomForestClassifier(501))

#fit the model on the data and predict the values

RFmodel=RFmodel.fit(x_train,y_train)

y_pred= RFmodel.predict(x_test)

#%%
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 

#%%

'''boosting : 1st single bag will be generated with 60% data and 40% data will be for testing purpose
after prediction we will take all the miss classifications values for bag 2on priority with extra data sum
total 60% for bag 2
it will be coninued upto the no provided for no of trees
 at the end the lass bag will be considered.'''
 
 '''adaboost algorithm : 
 gradient boost algorithm
 xgboost  algorithm'''
 
 #%%
 
 #using adaboost classifier
 
 from sklearn.ensemble import AdaBoostClassifier
 
 model_adaboost=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=50)) #50 is default
 
 model_adaboost.fit(x_train,y_train)
 y_pred=model_adaboost.predict(x_test)
 
 #%%
 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 

#%%
?GradientBoostingClassifier

#%%
 #using gradient classifier

 from sklearn.ensemble import GradientBoostingClassifier
 
 model_gradient=(GradientBoostingClassifier()) #100 is default
 
 model_gradient.fit(x_train,y_train)
 y_pred=model_gradient.predict(x_test)
 
 #%%
 from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 
 





