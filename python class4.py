# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 16:39:39 2018

@author: suvanjeet
"""
#%%
import pandas as pd
import numpy as np

pd.set_option('display.max_columns',None)    # print all columns
pd.set_option('display.max_rows',None)       # print all rows 
#adult=pd.read_csv(r'D:\Python day executions\files\adult_data.csv',header=None) 
adult=pd.read_csv(r'D:\Python day executions\files\adult_data.csv',header=None, delimiter=' *, *', engine='python') 

print(adult.head())
#print(adult.shape)
#%%
adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']   # head provided as head is not there

print(adult.head())
print(adult.isnull().sum())
#%%
adult=adult.replace(['?'],np.nan)   # to replace ? to nan in categorical field
print(adult.isnull().sum())
#%%
#create a copy of a dataframe
adult_df=pd.DataFrame.copy(adult)
adult_df.describe(include='all')
#%%
# top value will replaced with nan values
for value in ['workclass','occupation','native_country']:
    adult_df[value].fillna(adult_df[value].mode()[0],inplace=True)

print(adult_df.isnull().sum()) 
#%%
adult_df.workclass.value_counts()  # will create the value counts in workclass
#to select the categotrical colmns to change the values to neumerical
print(adult_df.dtypes)
#%%
columns = ['workclass', 'education',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income'] 
print(columns)
#%%
 # for preprocessing the data

from sklearn import preprocessing

le={}     # dictinary as it will be in key and value pair


for x in columns:
    le[x]=preprocessing.LabelEncoder()   # will lebel all categorical var in columns
    
for x in columns:
    adult_df[x]=le[x].fit_transform(adult_df[x])    # will replace the label with the variable data labeled
adult_df.describe(include='all')
    
#%%
print(adult_df.head())   
print(adult_df[['workclass', 'education',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']].head())
    
x= adult_df.values[:,:-1]    # includes the columns other than income
y= adult_df.values[:,-1]     # includes the income column
print(x.dtype)


#%%

import statsmodels.api as sm
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())
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
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())

#fitting training data in the model
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
print(list(zip(y_test,y_pred)))

classifier.score(x_test, y_test)

#%%

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(y_test,y_pred)
print(cfm)                              # to print confusion matrix , classification details

print("classification report")

print(classification_report(y_test,y_pred))  #recall - howmuch % +ve tc did we catch
'''class 1= TP/(TP+FN)= 1028/1028+1318=0.438      Class 0= TN/( TN+FP)=7009/7009+414= 0.9442

sensitivity and specitivity value should be closer to 1 then good model
precision = how much % of the +ve predictions are correct
class1 =TP/TP+FP = 1028/1028+414 = 0.71
Class 0 = TN/TN+FN = 7009/7009+1318  =0.84
F1-score = 2* precision*recall/(precision+recall)   closer to 1 then good

FP are type 1 error and FN are type 2 error

The 1st motive is to reduce type 2 error as it is not acceptabel in industry
'''

acc= accuracy_score(y_test,y_pred)
print("accuracy of the model: ", acc)   # to print the accuracy

#%%
# store the predicted probabilities
y_pred_prob= classifier.predict_proba(x_test)
print(y_pred_prob)

#%%
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value >0.35:                 #it has taken the weightage for col 1 in y_pred_prob
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

#print(y_pred_class)

#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value >0.6:                  #it has taken the weightage for col 0 in y_pred_prob
        y_pred_class.append(0)
    else:
        y_pred_class.append(1)

#print(y_pred_class)

#%%

cfm= confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm)
acc= accuracy_score(y_test,y_pred_class)
print("accuracy of the model: ", acc)

print(classification_report(y_test,y_pred_class)) 

#%%

# for loopfor tuning

for  a in np.arange(0,1,0.05):
    predict_mine= np.where(y_pred_prob[:,1]>a,1,0)
    cfm= confusion_matrix(y_test.tolist(),predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print('Errors at threshould',round(a,2)," : ", total_err, ", type 2 error-",cfm[1,0], ", type 1 error: ",cfm[0,1])

# 0.45 is good as errors are low and the type 2 error are also less

#%%
    # now set 0.45 as cut off
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value >0.45:                 #it has taken the weightage for col 1 in y_pred_prob
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

#print(y_pred_class)
        
# and create confusion matrix
        
cfm= confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm)
acc= accuracy_score(y_test,y_pred_class)
print("accuracy of the model: ", acc)

print(classification_report(y_test,y_pred_class)) 

#%%
from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(y_test.tolist(), y_pred_class)
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)
print(threshold)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()

#%%
from sklearn import metrics

fpr, tpr, threshold = metrics.roc_curve(y_test.tolist(), y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)

 

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()

#%%

#cross validation

'''types of cross validations
k fold
repeated  k fold
leave one out
leave p out    : p no of obs. as testing and remaining as training
stratified k fold:  maintains the ratio in distribution in each k fold section  '''
  
#%%

#Using cross validation

classifier=(LogisticRegression())

from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(x_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=x_train,y=y_train, cv=kfold_cv)
print(kfold_cv_result)
#%%
#finding the mean
print(kfold_cv_result.mean())

for train_value, test_value in kfold_cv:
    classifier.fit(x_train[train_value], y_train[train_value]).predict(x_train[test_value])


y_pred=classifier.predict(x_test)
#print(list(zip(Y_test,Y_pred)))

#%%
        
cfm1= confusion_matrix(y_test,y_pred)
print(cfm1)
acc= accuracy_score(y_test,y_pred)
print("accuracy of the model: ", acc)

print(classification_report(y_test,y_pred)) 

#%%
''' based on variance : variance threshould - to remove the minimal value in column
recursive feature elimation : variable importance calculation RFE(classifier, no of var to keep)
univariate  feature selection :measure chisquare value( selectkbest and selectpercentile)
    selectkbest

'''
#%%
# FEATURE SELECTION USING RFE SELECTION
x= adult_df.values[:,:-1]    # includes the columns other than income
y= adult_df.values[:,-1]     # includes the income column
print(x.dtype)
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
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())

#%%
colname=adult_df.columns[:]

#%%

from sklearn.feature_selection import RFE
rfe= RFE(classifier,7)
model_rfe=rfe.fit(x_train,y_train)  
print("num features: " , model_rfe.n_features_)    # will show the no of vars selected (7)
print('selected features: ')
print(list(zip(colname, model_rfe.support_)))     # will show the column names with true value as taken and false vale as not taken
print('feature ranking: ', model_rfe.ranking_) # will show 1 value as column selected and highest value as columns eliminated 1st and so on

#%%

y_pred=model_rfe.predict(x_test)

#%%
cfm1= confusion_matrix(y_test,y_pred)
print(cfm1)
acc= accuracy_score(y_test,y_pred)
print("accuracy of the model: ", acc)

print(classification_report(y_test,y_pred)) # this model is not good as rfe is randomly selected the columns 
# the better practice is to keep columns which is required as per the domain knowledge and then apply RFE

#%%
# FEATURE SELECTION USING UNIVARIATE SELECTION

x= adult_df.values[:,:-1]    # includes the columns other than income
y= adult_df.values[:,-1]     # includes the income column

#%%
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=8)    # mentioned the value as the no of variables to keep
fit1 = test.fit(x, y)

print(fit1.scores_)       # chi2 value for all variables
print(list(zip(colname,fit1.get_support())))   # will show the column names with true value as taken and false vale as not taken
features = fit1.transform(x)  # will store the 8 variables


print(features)

#%%
from sklearn.preprocessing import  StandardScaler
scaler= StandardScaler()
scaler.fit(features)       # only for train data

x= scaler.transform(features)  # for test and train both
print(x)

#%%
#splitting the data in test and train
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=10)

#%%
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression())
#fitting training data in the model
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)


#%%
cfm1= confusion_matrix(y_test,y_pred)
print(cfm1)
acc= accuracy_score(y_test,y_pred)
print("accuracy of the model: ", acc)

print(classification_report(y_test,y_pred))

#%%

#outlier detection

adult=pd.read_csv(r'D:\Python day executions\files\adult_data.csv',header=None, delimiter=' *, *', engine='python')
print(adult.head())


#%%
adult.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income'] 

#%%

import matplotlib.pyplot as plt
adult.boxplot()
plt.show()

#%%


adult.boxplot(column= 'fnlwgt')
plt.show()

#%%
adult.boxplot(column= 'age')
plt.show()

#%%

q1=adult['age'].quantile(0.25)
q3=adult['age'].quantile(0.75)
iqr= q3-q1

low=q1-1.5*iqr
high=q3+1.5*iqr

print(high,"    ", low)
#%%

adult_include= adult.loc[(adult['age'] >= low) & (adult['age'] <= high) ]  # meeting the range  loc is to refer the data frame
adult_exclude= adult.loc[(adult['age'] < low) | (adult['age'] > high) ]  # not meeting the range

#print("included ", adult_include , "  Excluded ",adult_exclude )


#%%
print(adult_include.shape)
print(adult_exclude.shape)

''' capping approach - value falling below lower wishker(min) value will be assigned with min value(same for max)
central-tendency approach - will assign the mean value to the outlier value


as age min value is -2 here so we are proceeding with central tendency'''

#%%
# central tendency approach
age_mean= int(adult_include.age.mean())   # mean of acceptable range
print(age_mean)

#%%
#importing  outlier values with mean value

adult_exclude.age=age_mean
adult_exclude.shape

#%%

adult_rev= pd.concat([adult_include, adult_exclude])

adult_rev.shape


#%%

#capping approach (alternate)
'''adult_exclude.loc[adult_exclude['age'] <low, 'age'] = low
adult_exclude.loc[adult_exclude['age'] >high, 'age'] = high'''

#%%



























































    
    