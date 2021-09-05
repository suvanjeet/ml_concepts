#%%
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data = pd.read_csv(r'D:\Python day executions\Dataset\Dataset.csv',
                 sep=",", header= 0)
data.head()
#%%
data.shape
#%%
data['target1'].value_counts()
#%%

data1=data[data.target1!=' ']
#%%
data1['target1'].value_counts()
#%%
data1.i_geocode_poi_km05__clinic.dtype
#%%
data1.target1.dtype
#%%
data1['target1']=data1.target1.astype(int)
#%%
data1.head()
#%%
for c in data1.columns: 
    if data1[c].dtype != 'int32':
        print(c)
#%%
data1.corr()
#%%
cor=data1.corr()['target1']
#%%
cor.sort_values(ascending=False).tail(20)

#%%
data1.isnull().sum()
#%%
data1['i_geocode_poi_km10_mã…yn'].value_counts()
#%%
cor1=pd.DataFrame(cor,index=None)
cor1.head()
#%%
cor1.target1.dtype
#%%
cor1=cor1.reset_index()
cor1.head()
#%%
cor2=cor1[cor1.target1.notnull()]
cor2.head()

#%%
col_remain=[]
for c in range(0,len(cor2)):
    col_remain.append(cor2.iloc[c]['index'])
col_remain
#%%
data2= data1[col_remain]
data2.head()
#%%
data2.shape


#%%
data2.drop_duplicates(subset=None,keep='first',inplace=True)
data2.head()
#%%
col_remain2=[]
for i in range(0,len(cor2)):
    if cor2.iloc[i]['target1'] < (-0.025) or cor2.iloc[i]['target1'] > 0.025:
            col_remain2.append(cor2.iloc[i]['index'])
col_remain2
#%%          
len(col_remain2)
#%%
data3= data1[col_remain2]
data3.head()
#%%
data3.shape
#%%
data3=data3.drop(['ExternalId'],axis=1)

#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
ind_df=data3.iloc[:,:-1]
ind_df.shape[1]
vif_df = pd.DataFrame()
vif_df.shape[1]
vif_df["features"] = ind_df.columns
vif_df["VIF Factor"] = [vif(ind_df.values, i) for i in range(ind_df.shape[1])]
print(vif_df)
#%%
data3=data3.drop(['i_geocode_poi_km10_school'],axis=1)
#%%
data3=data3.drop(['i_geocode_poi_km10_fuel'],axis=1)
#%%
data3=data3.drop(['i_geocode_poi_km10_post_office'],axis=1)
#%%
data3=data3.drop(['i_geocode_poi_km10_parking'],axis=1)
#%%
#%%
data3=data3.drop(['i_geocode_poi_km10_pub'],axis=1)
#%%
data3=data3.drop(['i_geocode_poi_km10_kindergarten'],axis=1)
#%%
x= data3.values[:,:-1]    # includes the columns other than income
y= data3.values[:,-1]     # includes the income column
print(x.dtype)
#%%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=10)

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
#%%
x_train
#%%

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 200, random_state=10,bootstrap=True)
model=classifier.fit(x_train, y_train)   
#%%
y_pred = model.predict(x_train) 

#%%
from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred))
cfm=confusion_matrix(y_train,y_pred)
print(cfm) # to print confusion matrix , classification details
#%%
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators= 200, random_state=10)
model=classifier.fit(x_train, y_train)
y_pred = model.predict(x_train)
#%%
from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred))
cfm=confusion_matrix(y_train,y_pred)
print(cfm)

#%%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
model=classifier.fit(x_train, y_train)
y_pred = model.predict(x_train)
#%%
from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred))
cfm=confusion_matrix(y_train,y_pred)
print(cfm)
