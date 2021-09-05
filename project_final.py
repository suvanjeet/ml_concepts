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
#%%
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
sns.countplot(x='grade',hue='default_ind',data=df1)
#%%
sns.countplot(x='addr_state',hue='default_ind',data=df1)
#%%
sns.countplot(x='purpose',hue='default_ind',data=df1)
#%%
import seaborn as sns
sns.regplot(x=df1["default_ind"], y=df1["annual_inc"])
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
correlations_data = df.corr()['default_ind'].sort_values()
#Print the most negative correlations
print(correlations_data.head(50), '\n')
#%%
df_vif=df1.drop('issue_d',axis=1)
df_vif.head()

#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
ind_df=df_vif.iloc[:,:-1]
ind_df.shape[1]
vif_df = pd.DataFrame()
vif_df.shape[1]
vif_df["features"] = ind_df.columns
vif_df["VIF Factor"] = [vif(ind_df.values, i) for i in range(ind_df.shape[1])]
print(vif_df)
#%%
vif_df['VIF Factor'].max()
#%%
hi_vif = ['loan_amnt', 'total_pymnt', 'out_prncp_inv', 'funded_amnt', 
'total_pymnt_inv', 'policy_code', 'funded_amnt_inv','int_rate', 
'sub_grade', 'installment', 'open_acc', 'last_credit_pull_d']
#hi_vif = ['']
df2 = df1.drop(hi_vif, axis=1)
df2.shape
#%%
df2['default_ind'].value_counts()
#%%
df2.groupby('default_ind').mean()
#%%
list_boxplot = ['annual_inc','dti', 'delinq_2yrs', 'inq_last_6mths','open_acc',
             'pub_rec','revol_bal','revol_util','total_acc','out_prncp',
             'total_rec_prncp',
             'total_rec_int','total_rec_late_fee','recoveries',
             'collection_recovery_fee', 'last_pymnt_amnt', 
             'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt',
             'tot_cur_bal','total_rev_hi_lim']
#%%
df2.boxplot(column= 'annual_inc')
plt.show()
#%%
q1=df2['annual_inc'].quantile(0.25)
q3=df2['annual_inc'].quantile(0.75)
iqr= q3-q1

low=q1-1.5*iqr
high=q3+1.5*iqr

print(high,"    ", low)
#%%

df3= df2.loc[(df2['annual_inc'] >= low) & (df2['annual_inc'] <= high) ]  
'''adult_exclude= df2.loc[(df2['annual_inc'] < low) | (df2['annual_inc'] > high) ]  # not meeting the range

#print("included ", adult_include , "  Excluded ",adult_exclude )'''
#%%
df3.boxplot(column= 'dti')
plt.show()
#%%
q1=df3['dti'].quantile(0.25)
q3=df3['dti'].quantile(0.75)
iqr= q3-q1

low=q1-1.5*iqr
high=q3+1.5*iqr

print(high,"    ", low)
#%%
df4= df3.loc[(df3['dti'] >= low) & (df3['dti'] <= high) ]  
#%%
df4.boxplot(column= 'total_rec_prncp')
plt.show()
#%%
q1=df4['total_rec_prncp'].quantile(0.25)
q3=df4['total_rec_prncp'].quantile(0.75)
iqr= q3-q1

low=q1-1.5*iqr
high=q3+1.5*iqr

print(high,"    ", low)
#%%
df5= df4.loc[(df3['total_rec_prncp'] >= low) & (df4['total_rec_prncp'] <= high) ]

#%%
sns.heatmap(df5)
#%%
import seaborn as sns
sns.countplot(x='default_ind',data=df5, palette='hls')
sns.plt.show()
plt.savefig('count_plot')
#%%
import seaborn as sns
sns.regplot(x=df5["default_ind"], y=df5["annual_inc"])
sns.plt.show()

#%%
sns.regplot(x=df5["default_ind"], y=df5["application_type"])
sns.plt.show()
#%%

sns.regplot(x=df5["default_ind"], y=df5["emp_length"])
sns.plt.show()

#%%
sns.heatmap(df5, annot=True, annot_kws={"size": 7})
#%%
# June 2007 to May 2015 - training data, from June 2015 to Dec 2015 Testing data
# June 2007 to May 2015 - training data, from June 2015 to Dec 2015 Testing data
test_month = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']
Test_data = df5.loc[df5['issue_d'].isin(test_month)]
Train_data = df5.loc[~df5['issue_d'].isin(test_month)]
#%%
Test_data = Test_data.drop('issue_d', axis=1)
Train_data = Train_data.drop('issue_d', axis=1)

Test_data.shape[0]+Train_data.shape[0]
df5.shape[0]
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
lr_acc = fit_and_evaluate(lr)
print('Logistic Regression Classifier on the test set: ACC = %0.4f' % lr_acc)
#%%
random_forest = RandomForestClassifier(n_estimators= 100, random_state=10)
random_forest_acc = fit_and_evaluate(random_forest)
print('Random Forest Classifier Performance on the test set: ACC = %0.6f' % random_forest_acc)
#%%
gradient_boosted = GradientBoostingClassifier(n_estimators=100, random_state=10)
gradient_boosted_acc = fit_and_evaluate(gradient_boosted)
print('Gradient Boosted Classifier Performance on the test set: ACC = %0.6f' % gradient_boosted_acc)
#%%
gradient_boosted = GradientBoostingClassifier(n_estimators=200, random_state=10)
gradient_boosted_acc = fit_and_evaluate(gradient_boosted)
print('Gradient Boosted Classifier Performance on the test set: ACC = %0.6f' % gradient_boosted_acc)
#%%
#plt.style.use('fivethirtyeight')
#figsize(8, 6)
# Dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Logistic Regression', 'Random Forest', 'Gradient Boosted'],
'acc': [lr_acc,random_forest_acc,gradient_boosted_acc]})
# Horizontal bar chart of test accuracy
model_comparison.sort_values('acc', ascending = False).plot(x = 'model', y = 'acc', kind = 'barh',
color = 'green', edgecolor = 'black')
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
# Plot formatting
plt.ylabel(''); plt.yticks(size = 14); plt.xlabel('Accuracy'); plt.xticks(size = 12)
plt.title('Model Comparison on Test ACC', size = 20);


#%%
'''from sklearn.metrics import classification_report
model1 = lr.fit(X_train, y_train)
y_pred = model1.predict(X_test)
classification_report(y_test, y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm) # to print confusion matrix , classification details
y_pred_prob= lr.predict_proba(X_test)
print(y_pred_prob)'''
#%%
from sklearn.metrics import classification_report
model2 = gradient_boosted.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
classification_report(y_test, y_pred2)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(y_test,y_pred2)
print(cfm) # to print confusion matrix , classification details
#%%
y_pred_prob= gradient_boosted.predict_proba(X_test)
print(y_pred_prob)
#%%
y_pred1 = model1.predict(X_train)
cfm=confusion_matrix(y_train,y_pred1)
print(cfm)
#%%
print("classification report")
print(classification_report(y_test,y_pred))

#%%
classifier=(GradientBoostingClassifier())
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=5)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,y=y_train, cv=kfold_cv)
print(kfold_cv_result)
#%%
#finding the mean
print(kfold_cv_result.mean())
#%%
plt.figure(figsize=(9,9))
sns.heatmap(cfm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
#%%
for  a in np.arange(0,1,0.05):
    predict_mine= np.where(y_pred_prob[:,1]>a,1,0)
    cfm= confusion_matrix(y_test,predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print('Errors at threshould',round(a,2)," : ", total_err, ", type 2 error-",cfm[1,0], ", type 1 error: ",cfm[0,1])
#%%
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value < 0.3:                 
        y_pred_class.append(0)
    else:
        y_pred_class.append(1)
#%%       
model_acc = gradient_boosted.score(y_test, y_pred_class)
model_acc
#%%
gradient_boosted.feature_importances_
#%%
df5.columns
#%%
model_acc = gradient_boosted.score(y_test, y_pred_class)
model_acc
#%%
cfm=confusion_matrix(y_test,y_pred_class)
print(cfm)
#%%
print("classification report")
print(classification_report(y_test,y_pred_class))
#%%
from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_class)
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
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_class)
auc = metrics.auc(fpr,tpr)
print(auc)

