# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 20:59:41 2018

@author: suvanjeet
"""

#%%
import pandas as pd
import numpy as np

pd.set_option('display.max_columns',None)    # print all columns
pd.set_option('display.max_rows',None)       # print all rows 
#%%
#adult=pd.read_csv(r'D:\Python day executions\files\adult_data.csv',header=None) 
'''train_data=pd.read_csv(r'D:\Python day executions\files\riskanalytics\risk_analytics_train.csv',header=0) 
test_data=pd.read_csv(r'D:\Python day executions\files\riskanalytics\risk_analytics_test.csv',header=0) 

print(train_data.head())
print(test_data.head())'''
#%%
df = pd.read_csv(r'C:\Users\suvanjeet\Desktop\python project\XYZCorp_LendingData.txt', sep="\t", header= 0)
#%%
df.head()
#%%
df.columns
#%%
df.shape
#%%

df.describe(include='all')
#%%
print(df.isnull().any(axis=1).sum())
#%%
print(df.isnull().sum())

#%%

col1= ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
       'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title',
       'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
       'issue_d', 'pymnt_plan', 'desc', 'purpose', 'title', 'zip_code',
       'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line',
       'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
       'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
       'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt',
       'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
       'collections_12_mths_ex_med', 'mths_since_last_major_derog',
       'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
       'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt',
       'tot_cur_bal', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
       'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
       'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util',
       'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
       'default_ind']
#%%
df['emp_title']
#%%
col2 =['id','member_id','emp_title','desc','title']

'term' 'grade'  'sub_grade' 'emp_length' 'home_ownership' 'verification_status'
'pymnt_plan' 'purpose' 'zip_code' 'addr_state'
#%%
ar= df['issue_d'].value_counts()
 #%%
 
 col_drop = ['id','member_id']
 df.drop(col_drop,axis=1,inplace=True)
 
 #%%
 
 per_miss=df.isnull().sum()/ len(df)
 per_miss
 
 #%%
 
 missing_feature=per_miss[per_miss>0.5].index
 missing_feature
 #%%
 
 df1=df.drop(missing_feature,axis=1)
 df1.shape
 
 #%%
 df1.columns
 
 #%%
 df1.isnull().sum()
 
 #%%
 
 miss_mode = ['term', 'grade', 'sub_grade', 'emp_title', 'emp_length', 
              'home_ownership','verification_status','issue_d', 'pymnt_plan',
              'purpose','title', 'zip_code','addr_state', 'earliest_cr_line', 
              'initial_list_status','last_pymnt_d', 'next_pymnt_d','last_credit_pull_d',
              'policy_code','application_type', 'int_rate'] 
 miss_mean = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv','installment', 'annual_inc',
              'dti', 'delinq_2yrs', 'inq_last_6mths','open_acc', 'pub_rec','revol_bal',
              'revol_util','total_acc','out_prncp', 'out_prncp_inv','total_pymnt',
              'total_pymnt_inv','total_rec_prncp', 'total_rec_int','total_rec_late_fee',
              'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 
              'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt', 
              'tot_cur_bal','total_rev_hi_lim'] 
 print(len(miss_mode) )
 len(miss_mean)
 
 #%%
 for i in miss_mode:
    df1[i].fillna(df1[i].mode()[0],inplace=True)
#%%
 for i in miss_mean:
    df1[i].fillna(df1[i].mean(),inplace=True)
 
 #%%
 df1.info()
 #%%
 df1.isnull().sum()
 #%%
 df1.dtypes
 
 #%%
 col_obj=df1.select_dtypes('object').columns
 col_obj
 
 #%%
 col_obj=col_obj.drop('issue_d')
#%%
 col_obj
 
 #%%
 from sklearn import preprocessing

le={}

for a in col_obj:
    le[a]= preprocessing.LabelEncoder()
for a in col_obj:
    df1[a]= le[a].fit_transform(df1[a])
 #%%
 
 df1.head()

 
 #%%
 
 from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
ind_df=df1.iloc[:,:-1]
ind_df.shape[1]
vif_df = pd.DataFrame()
vif_df.shape[1]
vif_df["features"] = ind_df.columns
vif_df["VIF Factor"] = [vif(ind_df.values, i) for i in range(ind_df.shape[1])]
print(vif_df)

#%%
hi_vif = ['loan_amnt', 'total_pymnt', 'out_prncp_inv', 'funded_amnt', 
'total_pymnt_inv', 'policy_code', 'funded_amnt_inv','int_rate', 
'sub_grade', 'installment', 'open_acc', 'last_credit_pull_d']
#hi_vif = ['']
df2=df1.drop(hi_vif, axis=1)

#%%

test_month = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']
Test_data = df2.loc[df2['issue_d'].isin(test_month)]
Train_data = df2.loc[~df2['issue_d'].isin(test_month)]
#%%
Test_data=Test_data.drop('issue_d',axis=1)
Train_data=Train_data.drop('issue_d',axis=1)

#%%

Test_data.shape[0]+Train_data.shape[0]
df.shape[0]

#%%
x_train= pd.DataFrame (Train_data.values[:,:-1] )   # includes the columns other than income
y_train= pd.DataFrame(Train_data.values[:,-1] )

x_test= pd.DataFrame(Test_data.values[:,:-1])    # includes the columns other than income
y_test= pd.DataFrame (Test_data.values[:,-1] )

#%%
x_train.head()

#%%

from sklearn.preprocessing import  StandardScaler
scaler= StandardScaler()
scaler.fit(x_train)       # only for train data

x_train= scaler.transform(x_train)  # for test and train both

#%%

from sklearn.preprocessing import  StandardScaler
scaler= StandardScaler()
scaler.fit(x_test)       # only for train data

x_test= scaler.transform(x_test)  # for test and train both

#%%

