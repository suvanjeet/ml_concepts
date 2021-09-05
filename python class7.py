# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 14:17:53 2018

@author: suvanjeet
"""

#%%
#ensemble modelling
''' no bagconcept, entire training data is used for different models separately
for test it will go through all the models and predict the mejority output from them
votingclassifier(all models sep by comma)'''

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#%%
#create the sub models

estimators=[]
'''model1 = LogisticRegression()
estimators.append(('log',model1))'''
model2 = DecisionTreeClassifier()
estimators.append(('cart',model2))
model3 = SVC(kernel='rbf',C=15,gamma=0.6)
estimators.append(('svm',model3))

print(estimators)

#%%
# create the ensemble model

ensemble= VotingClassifier(estimators)
ensemble.fit(x_train,y_train)
y_pred=ensemble.predict(x_test)
print(y_pred)

#%%
 from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 
 
#%%
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
#%%
url='http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

names=['Sample code number',
    'Clump Thickness',               
    'Uniformity of Cell Size',    
    'Uniformity of Cell Shape',    
    'Marginal Adhesion',           
    'Single Epithelial Cell Size', 
    'Bare Nuclei',      
    'Bland Chromatin',     
    'Normal Nucleoli',              
    'Mitoses',                    
    'Class'] 

'''url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
name=["ID number","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size",
"Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]'''

df=pandas.read_csv(url,names=names)
df.head()

#%%

df.info()
#%%
df.isnull().sum()

#%%
df=df.replace(['?'],np.nan)

df.isnull().sum()

#%%
df.describe(include='all')

#%%

df['Bare Nuclei'].fillna(df['Bare Nuclei'].mode()[0],inplace=True)
df.isnull().sum()

#%%
x=df.values[:,1:-1]
y=df.values[:,-1]
#%%

df.dtypes

#%%
df['Bare Nuclei']=df['Bare Nuclei'].astype(int)
df.dtypes

#%%

x=df.values[:,1:-1]
y=df.values[:,-1]

#%%

#splitting the data in test and train
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=10)


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
# store the predicted probabilities
y_pred_prob= classifier.predict_proba(x_test)
#print(y_pred_prob)
y_pred_prob=np.round(y_pred_prob,2)
print(y_pred_prob)


#%%
# for loopfor tuning

for  a in np.arange(0,1,0.05):
    predict_mine= np.where(y_pred_prob[:,1]>a,4,2)  # 4, 2 because dependent variable have output 4 and 2
    cfm= confusion_matrix(y_test.tolist(),predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print('Errors at threshould',round(a,2)," : ", total_err, ", type 2 error-",cfm[1,0],
          ", type 1 error: ",cfm[0,1])

#%%
    
    #ensemble model
estimators=[]
'''model1 = LogisticRegression()
estimators.append(('log',model1))'''
model2 = DecisionTreeClassifier()
estimators.append(('cart',model2))
model3 = SVC()
estimators.append(('svm',model3))

print(estimators)

#%%
# create the ensemble model

ensemble= VotingClassifier(estimators)
ensemble.fit(x_train,y_train)
y_pred=ensemble.predict(x_test)
print(y_pred)

#%%
 from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details
print(accuracy_score(y_test,y_pred)) 
print("classification report")

print(classification_report(y_test,y_pred)) 

#%%











