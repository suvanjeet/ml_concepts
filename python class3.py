#%%

import pandas as pd

#%%

import numpy as np
data= np.array(['101','102','103','104'])
s=pd.Series(data)       #  it will assign the sl no automatically with all elements
print(s)
print()
print(s[1])

s=pd.Series(data,index=[1001,1002,1003,1004])  # to change the index or sl no, entries should match with no of element
print(s)
print(s[1001])

#%%
data= {'a':0.,'b':1.,'c':2.}
s=pd.Series(data)
print(s['a'])
print(s[-2:])

#%%

data=[['amit',40],['nikita',23],['clara',33]]
df=pd.DataFrame(data,columns=['name','age'])
print(df)

#%%

data=[['amit',40],['nikita',23],['clara',33,'mumb']]
df=pd.DataFrame(data,columns=['name','age'])       # will give error as 3rd column name not provided
df=pd.DataFrame(data,columns=['name','age',''])
df=pd.DataFrame(data,columns=['name','age','address']) # will print none if data not present
print(df)

#%%
data= {'name':['amit','nikita','clara'], 'age':[40,23,33]}
df=pd.DataFrame(data)
print (df)

df=pd.DataFrame(data,index=['rank1','rank2','rank3'],columns=['age ','name'])  #in order to configure the cloumn to print seq.
print(df)

#%%
data= {'name':['amit','nikita','clara'], 'age':[40,23,33]}
df=pd.DataFrame(data)
print (df)

df['address']=['mumbai','pune','mumbai']   # if address not present then it will add column to data set
print(df)


df=df[['name','address','age']]    # alternate way of rearrange coumns
print(df)


df['newcol']=5
print(df)

df['revisedcol']=df['newcol']*2    # modified col w.r.t. diff col
print(df)

del df['newcol']                    # deleting a col
print(df)

df=df.drop(1)                 # it will delete the row which has 1 value
print(df)
#df=df.drop('revised_col')      # as axis not present so it would not find any row label named revisedcol
df=df.drop('revisedcol',axis=1)   # as axis is given so it will search column label and if found delete it
print(df)
#%%

data= {'name':['amit','nikita','clara'], 'age':[40,23,33]}
df=pd.DataFrame(data)
print (df)

df=pd.DataFrame(data,index=['rank1','rank2','rank3'],columns=['age ','name'])  #in order to configure the cloumn to print seq.
print(df)

df['address']=['mumbai','pune','mumbai']   # if address not present then it will add column to data set
print(df)

#accessing data elements using indexes
#df.loc[inclusive:inclusive]

#print(df.loc['rank2'])
print(df.loc['rank2':'rank3'])

#df.iloc[inclusive:exclusive]
#print(df.iloc[0:1])

#print(df.loc['rank2':'rank3',['address','age']])      #subset
print(df.iloc[:,[1,2]])
#%%
data= {'name':['amit','nikita','clara'], 'age':[40,23,33]}
df=pd.DataFrame(data)
print (df)

df['address']=['mumbai','pune','mumbai']   # if address not present then it will add column to data set
print(df)

df[['name','age']]   # will ot affect the original data set its only for viewing


#%%
data= {'name':['amit','nikita','clara'], 'age':[40,23,33]}
df=pd.DataFrame(data)
print (df)

df['address']=['mumbai','pune','mumbai']   # if address not present then it will add column to data set
print(df)


#saving df to file in specified path  r is consider the "/" as a part of path
df.to_csv(r'C:\Users\suvanjeet\Desktop\sampledf.csv', index=True,header=True)  #index true means we want to save it in file 

df.to_excel(r'C:\Users\suvanjeet\Desktop\sampledf.xls',index=False,header=True)

print('done')

#%%

df2=pd.read_csv(r'C:\Users\suvanjeet\Desktop\sampledf.csv',index_col=0) 
 #index col 0 otherwise it will take it as one column
print(df2)

df3=pd.read_excel(r'C:\Users\suvanjeet\Desktop\sampledf.xls')
print(df3)

print(df2.dtypes)     # data types of all columns in table
print(df2.age.dtypes)  # type of the specified col
df2.info()             #returns column details no of obs, and missing values
print(df2.shape)        # returns dimensions

print(df2)
df2.set_value(2,['name','age'],['john',23])  #1st arg is row and 2nd arg is col and 3rd is the value want to assign
print(df2)

df2.loc[df2['name']=='nikita','address']='delhi'  # check the name is nikita then change the address to delhi
print(df2)

df2.loc[df2['name']=='suva','address']='delhi'  # check the name is nikita then change the address to delhi
print(df2)

df2.sort_values(['name'],ascending=False)    #sorting (here descending) if ascending=then true(by default)

df2.sort_index(axis=1)    # sort the columns as axis=1 capitals come 1st then lowercase

df2=df2
#%%

pd.set_option('display.max_columns',None)    # print all columns
pd.set_option('display.max_rows',None)       # print all rows 

titanic_df=pd.read_excel(r'D:\Python day executions\files\Titanic_Survival_Train.xls',index_col=0)

print(titanic_df)
print(titanic_df.head(7))
print(titanic_df.tail())

print(titanic_df.info())
print(titanic_df.dtypes)
print(titanic_df.shape)
print(titanic_df.describe())   # for integer variables
print(titanic_df.describe(include=[np.object])) # for object variables
# top=mode and freq= freq of mode value

print(titanic_df.describe(include='all'))    # for all variables

print(titanic_df.Sex.describe())     # for perticular variable
my_df= titanic_df[['Sex','Pclass','Age']]
print(type(my_df))      # data frame as passed as list

df_agemorethan60= titanic_df[titanic_df['Age']>60]
print(df_agemorethan60.shape)


#passengers whose age is more 60 and are male and survived
my_df= titanic_df[(titanic_df['Age']>60) &(titanic_df['Sex']=='male')& (titanic_df['Survived']==1)]

print(my_df.shape)
print(my_df.shape[0])    # for rows
print(my_df.shape[1])       # for columns


myseries=titanic_df['Pclass']
print(myseries.value_counts())   # for categorical variable

print(pd.crosstab(titanic_df['Sex'],titanic_df['Survived']))  # independent with dependant

print(titanic_df['Sex'][titanic_df['Survived']==1].value_counts())

# converting neumerical to category data

PassengerAge=titanic_df['Age']
PassengerAge=PassengerAge.dropna()    # frop na values
Bins=[0,15,21,60,PassengerAge.max()]   # it will create range 0-15, 16-21, 22-60,61-max
Binlabels=['children','adolescents','adults','seniors']   # label names for ranges
categories=pd.cut(PassengerAge,Bins,labels=Binlabels, include_lowest=True) # will take all inputs
print(categories.value_counts())   # count the values category wise

newdf= pd.concat([PassengerAge,categories], axis=1)
print(newdf.head())

print(titanic_df.isnull().sum())
print(titanic_df.isnull().any(axis=1).sum()) # will return the row conut if at leant any column holds null

print(titanic_df['Age'].isnull().sum())  # will return column wise


#print(titanic_df.dropna())  # will drop all rows having at least one column value null

# null will replace with 0
#print(titanic_df.fillna(0))


# to drop the variable having null values greater than 50% of whole data

half_count= len(titanic_df)/2
titanic_df= titanic_df.dropna(thresh=half_count,axis=1)
print(titanic_df.isnull().sum())
print(titanic_df.dtypes)


# handling missing variable

#titanic_df=titanic_df.drop('cabin', axis=1)
#print(titanic_df.head())

# handing missing values in embarked column


titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0],inplace=True)
# to access categorical mode value we have to pass index which is [0]

import math
print(titanic_df['Embarked'].mode())

print(titanic_df.isnull().sum())


age_mean= int(titanic_df.Age.mean())
titanic_df['Age'].fillna(age_mean,inplace=True)

print(titanic_df.isnull().sum())

#%%
pd.set_option('display.max_columns',None)    # print all columns
pd.set_option('display.max_rows',None)       # print all rows 

titanic_1=pd.read_excel(r'D:\Python day executions\files\Titanic_Survival_Train.xls',header=0,index_col=None)
print(titanic_1.head(7))
print(titanic_1.info())
print(titanic_1.describe(include='all'))
#%%

'''for x in titanic_1.columns:
    print(x.value_counts())'''
    
col_1= titanic_1['Age']
print(col_1.value_counts())
#%%
col_1= titanic_1['Cabin']
print(col_1.value_counts())
#%%
'''y=titanic_1.columns
print(y)
for x in titanic_1.columns:
    print(titanic_1.x.value_counts())'''
titanic_2= titanic_1[['Age','Cabin','Embarked']]
print(titanic_2.head())
titanic_2.apply(pd.value_counts)

#%%
titanic_2.describe(include='all')

#%%

half_count= len(titanic_1)/2
titanic_1= titanic_1.dropna(thresh=half_count,axis=1)
print(titanic_1.isnull().sum())
print(titanic_1.head())

#%%
titanic_1['Embarked'].fillna(titanic_1['Embarked'].mode()[0],inplace=True)
titanic_1['Age'].fillna(titanic_1['Age'].mean(),inplace=True)
print(titanic_1.isnull().sum())

#%%

titanic_3=titanic_1.drop(['PassengerId','Name','Ticket'], axis=1)
print(titanic_3.head())

#%%

le1={}

for x in titanic_3[['Sex','Embarked']]:
    le1[x]= preprocessing.LabelEncoder() 
for x in titanic_3[['Sex','Embarked']]:
    titanic_3[x]=le1[x].fit_transform(titanic_3[x])    # will replace the label with the variable data labeled
print(titanic_3.head())

#%%
x= titanic_3.values[:,1:]    # includes the columns other than income
y= titanic_3.values[:,0] 

scaler1= StandardScaler()
scaler1.fit(x)

x=scaler1.transform(x)
#%%
print(x)
print(titanic_3.head())

#%%

y=y.astype(int)
y.dtype

#%%

x_train1,x_test1,y_train1,y_test1= train_test_split(x,y,test_size=0.3,random_state=10)

#%%
lr1=(LogisticRegression())

#fitting training data in the model
lr1.fit(x_train1,y_train1)

y_pred1=lr1.predict(x_test1)

#%%
print(list(zip(y_test1,y_pred1)))

lr1.score(x_test1, y_test1)

#%%

cfm=confusion_matrix(y_test1,y_pred1)
print(cfm)                              # to print confusion matrix , classification details
#%%
print("classification report")

print(classification_report(y_test1,y_pred1))  #recall - howmuch % +ve tc did we catch

acc= accuracy_score(y_test1,y_pred1)
print("accuracy of the model: ", acc)   # to print the accuracy









