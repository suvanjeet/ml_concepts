# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:54:30 2018

@author: suvanjeet
"""

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
print('Linear Regression Classifier on the test set: ACC = %0.6f' % lr_acc)
#%%
from sklearn.metrics import classification_report
model1 = lr.fit(X_train, y_train)
y_pred = model1.predict(X_test)
classification_report(y_test, y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm) # to print confusion matrix , classification details
y_pred_prob= lr.predict_proba(X_test)
print(y_pred_prob)

#%%
print("classification report")

print(classification_report(y_test,y_pred))


#%%

plt.figure(figsize=(9,9))
sns.heatmap(cfm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
#%%
# store the predicted probabilities

'''y_pred_prob= lr.predict_proba(X_test)
print(y_pred_prob)
# for loopfor tuning'''

#%%

for  a in np.arange(0,1,0.05):
    predict_mine= np.where(y_pred_prob[:,1]>a,1,0)
    cfm= confusion_matrix(y_test,predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print('Errors at threshould',round(a,2)," : ", total_err, ", type 2 error-",cfm[1,0], ", type 1 error: ",cfm[0,1])


#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value >0.5:                  #it has taken the weightage for col 0 in y_pred_prob
        y_pred_class.append(0)
    else:
        y_pred_class.append(1)
# 0.45 is good as errors are low and the type 2 error are also less
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

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_prob[:,1])
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
'''plt.figure(figsize=(9,9))
sns.heatmap(cfm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);'''
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









