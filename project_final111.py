# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:22:43 2018

@author: suvanjeet
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc

df = pd.read_csv(r'C:\Users\suvanjeet\Desktop\python project\XYZCorp_LendingData.txt', sep="\t", header= 0)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#%%

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#%%
col_name = df.columns
df.shape
#%%6g
df.head()
col_drop = ['id','member_id']
df.drop(col_drop, axis=1, inplace=True)
#%%
df.head()
per_miss = df.isnull().sum()/len(df)
print(per_miss)
per_miss.shape
#%%
missing_features = per_miss[per_miss > 0.50].index
print(missing_features)
df1 = df.drop(missing_features, axis=1)
df1.shape
#%%
colname = df1.columns
len(colname)
miss_mode = ['term', 'grade', 'sub_grade', 'emp_title', 'emp_length',
             'home_ownership','verification_status','issue_d',
             'pymnt_plan','purpose','title', 'zip_code','addr_state', 
             'earliest_cr_line', 'initial_list_status','last_pymnt_d',
             'next_pymnt_d','last_credit_pull_d','policy_code','application_type',
             'int_rate']

miss_mean = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv','installment',
             'annual_inc','dti', 'delinq_2yrs', 'inq_last_6mths','open_acc',
             'pub_rec','revol_bal','revol_util','total_acc','out_prncp',
             'out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp',
             'total_rec_int','total_rec_late_fee','recoveries',
             'collection_recovery_fee', 'last_pymnt_amnt', 
             'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt',
             'tot_cur_bal','total_rev_hi_lim']

len(miss_mode)
len(miss_mean)
#%%
sns.heatmap(df1.isnull())
#%%
for i in miss_mode:
   df1[i].fillna(df1[i].mode()[0], inplace = True)

for i in miss_mean:
   df1[i].fillna(df1[i].mean(), inplace = True)
#%%
df1.info()
df1.isnull().sum()
#%%
df1.head()
#%%
df1.describe(include='all')
#%%
import seaborn as sns
sns.countplot(x='default_ind',data=df1, palette='hls')
sns.plt.show()
plt.savefig('count_plot')

#%%
sns.countplot(x='emp_length',hue='default_ind',data=df1)
#%%
'''sns.countplot(x='emp_title',hue='default_ind',data=df1)'''

sns.countplot(x='grade',hue='default_ind',data=df1)

#%%

sns.countplot(x='addr_state',hue='default_ind',data=df1)

#%%
sns.countplot(x='purpose',hue='default_ind',data=df1)
#%%
'''
%matplotlib inline
pd.crosstab(df1.emp_title,df1.default_ind).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('title')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')'''
#%%

'''sns.pairplot(df1)
sns.plt.show()'''

#%%
import seaborn as sns
sns.regplot(x=df1["default_ind"], y=df1["annual_inc"])
sns.plt.show()

#%%
sns.regplot(x=df1["default_ind"], y=df1["application_type"])
sns.plt.show()

#%%
col_obj = df1.select_dtypes('object').columns
print(col_obj)
#%%
col_obj = col_obj.drop('issue_d')
print(col_obj)
#%%
from sklearn import preprocessing
le = {}
for a in col_obj:
    le[a]= preprocessing.LabelEncoder()
for a in col_obj:
    df1[a] = le[a].fit_transform(df1[a])  

df1.head()
#%%
df1.dtypes
#%%
# Find all correlations and sort 
#correlations_data = df.corr()['default_ind'].sort_values()
# Print the most negative correlations
#print(correlations_data.head(50), '\n')

#import seaborn as sns
#sns.heatmap(df1.corr())
#plt.show()

#from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
#ind_df=df1.iloc[:,:-1]
#ind_df.shape[1]
#vif_df = pd.DataFrame()
#vif_df.shape[1]
#if_df["features"] = ind_df.columns
#vif_df["VIF Factor"] = [vif(ind_df.values, i) for i in range(ind_df.shape[1])]
#print(vif_df)

#vif_df['VIF Factor'].max()

hi_vif = ['loan_amnt', 'total_pymnt', 'out_prncp_inv', 'funded_amnt', 
'total_pymnt_inv', 'policy_code', 'funded_amnt_inv','int_rate', 
'sub_grade', 'installment', 'open_acc', 'last_credit_pull_d']
#hi_vif = ['']
df2 = df1.drop(hi_vif, axis=1)
df2.shape
#%%
df2.columns

#%%
df2['emp_length'].unique()

#%%

'''data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])'''
#%%
df2['default_ind'].value_counts()

#%%

df2.groupby('default_ind').mean()
#%%
import seaborn as sns
sns.countplot(x='default_ind',data=df2, palette='hls')
sns.plt.show()
plt.savefig('count_plot')
#%%
import seaborn as sns
sns.regplot(x=df2["default_ind"], y=df2["annual_inc"])
sns.plt.show()

#%%
sns.regplot(x=df2["default_ind"], y=df2["application_type"])
sns.plt.show()
#%%

sns.regplot(x=df2["default_ind"], y=df2["emp_length"])
sns.plt.show()

#%%
sns.heatmap(df2, annot=True, annot_kws={"size": 7})
#sns.plt.show()


#%%
# June 2007 to May 2015 - training data, from June 2015 to Dec 2015 Testing data
# June 2007 to May 2015 - training data, from June 2015 to Dec 2015 Testing data
test_month = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']
Test_data = df2.loc[df2['issue_d'].isin(test_month)]
Train_data = df2.loc[~df2['issue_d'].isin(test_month)]
#%%
Test_data = Test_data.drop('issue_d', axis=1)
Train_data = Train_data.drop('issue_d', axis=1)

Test_data.shape[0]+Train_data.shape[0]
df.shape[0]
#%%
X_train= pd.DataFrame(Train_data.values[:,:-1])
y_train = pd.DataFrame(Train_data.values[:,-1])

X_test= pd.DataFrame(Test_data.values[:,:-1])
y_test = pd.DataFrame(Test_data.values[:,-1])

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 10)
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def fit_and_evaluate(model):   
    # Train the model
    model.fit(X_train, y_train)   
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_acc = model.score(X_test, y_test)   
    # Return the performance metric
    return model_acc

lr = LogisticRegression()
'''lr = RandomForestClassifier()
lr = GradientBoostingClassifier()'''
lr_acc = fit_and_evaluate(lr)
print('Linear Regression Classifier on the test set: ACC = %0.4f' % lr_acc)
#%%
from sklearn.metrics import classification_report
model1 = lr.fit(X_train, y_train)
y_pred = model1.predict(X_test)
classification_report(y_test, y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm) # to print confusion matrix , classification details


#%%
plt.figure(figsize=(9,9))
sns.heatmap(cfm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
#%%
classifier=(LogisticRegression())

from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=y_train, cv=kfold_cv)
print(kfold_cv_result)

#%%
#finding the mean
print(kfold_cv_result.mean())
#%%
''' heat map before vif(multi colinearity)
grade - default ind
emp_title default ind
addr_state - default ind
purpose - default'''