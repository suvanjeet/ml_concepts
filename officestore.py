#%%
import pandas as pd
import numpy as np
WS=pd.read_csv(r'C:\Users\suvanjeet\Desktop\timeseries\Walmart-stores.csv',header=0) 
print(WS)

#%%
WS.rename({'Store':'store'}, axis=1, inplace=True)
WS.head()

#%%
WS.head()
#%%
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(WS['store'], WS['Size'], s=None, c='red')
#%%
import seaborn as sns
#sns.scatterplot(x=Store, y=Size, data= WS , palette ='blue')
sns.regplot(x=WS["store"], y=WS["Size"], fit_reg=True)
sns.plt.show()
#%%
sns.countplot(x="Type", data=WS)

#%%
WT=pd.read_csv(r'C:\Users\suvanjeet\Desktop\timeseries\Walmart-train.csv',header=0) 

#%%
WT.head()
#%%

'''WT['type']='NA'
WT['size']='NA'
WT.head()'''

#%%
'''for i in range (0, len(WS)):
    for j in range(0,len(WT)):
        if WS.iloc[i]['store'] == WT.iloc[j]['Store']:
            WT.iloc[j]['type']=WS.iloc[i]['Type']
            WT.iloc[j]['size']=WS.iloc[i]['Size']
WT.head()'''
#%%
df= pd.merge(WT,WS[['store','Type','Size']], left_on='Store',right_on='store',how='left')
df

#%%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#%%
df=df.drop(['store'],axis=1)
#%%
df.head

#%%
df[df['Weekly_Sales'] > df['Size']].index.tolist()

#%%
df.index[round(df['Weekly_Sales']) > df['Size']].tolist()
#%%

df.iloc[149247]

#%%
store_selected = input("What is your store? ")
print ("Your store selected: ", store_selected)
store_selected= int(store_selected)

#%%
df1= df[df['Store'] == store_selected]
df1.head()
#%%
dept_selected = input("What is your dept? ")
print ("Your dept selected: ", dept_selected)
dept_selected= int(dept_selected)

#%%

df1= df[df['Dept'] == dept_selected]
#%%
df1.head()

#%%
len(df1)

#%%
df1.dtypes
#%%
df1['Date'] = pd.to_datetime(df1['Date'])

#%%
df1['Year'] = df1['Date'].dt.year
df1['Month'] = df1['Date'].dt.month
df1['Day'] = df1['Date'].dt.day

#%%
df1.head()
#%%
df2=df1.drop(['Store','Dept','Date','Size'], axis=1)
df2.head()
#%%
from sklearn import preprocessing
le = {}
le['IsHoliday']= preprocessing.LabelEncoder()
le['Type']= preprocessing.LabelEncoder()
df2['IsHoliday'] = le['IsHoliday'].fit_transform(df2['IsHoliday'])
df2['Type'] = le['Type'].fit_transform(df2['Type'])

#%%
df2.head()
#%%
len(df2)
#%%
total= round(len(df2)*0.7)
total
#%%
train_data=df2[0:total]
len(train_data)

#%%
test_data=df2[total:(len(df2))+1]
len(test_data)

#%%
train_data.head()
#%%
x_train=train_data.drop(['Weekly_Sales'], axis=1)
x_train.head()
#%%
y_train= train_data['Weekly_Sales']
y_train.head()
#%%
x_test=test_data.drop(['Weekly_Sales'], axis=1)
y_test=test_data['Weekly_Sales']
#%%
x_test.head()
#%%
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score 
  
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)
#%%
# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

#%%
import matplotlib.pyplot as plt 
## plotting residual errors in test data 
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, 
            color = "blue", s = 10, label = 'Test data') 
  
## plotting line for zero residual error 
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2) 
#%% 
# variance score: 1 means perfect prediction 
print('Variance score: {}'.format(reg.score(x_test, y_test))) 
#%% 
# plot for residual error 
  
## setting plot style 
plt.style.use('fivethirtyeight') 





































