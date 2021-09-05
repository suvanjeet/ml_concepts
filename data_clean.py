# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:17:51 2018

@author: suvanjeet
"""
#%%
import pandas as pd

pd.set_option('display.max_columns',None)    # print all columns
pd.set_option('display.max_rows',None)       # print all rows 

pub_df=pd.read_csv(r'C:\Users\suvanjeet\Desktop\BL-Flickr-Images-Book.csv',index_col=0) 
pub_df
print(pub_df.head())

print(pub_df.info())

droplist = ['Edition Statement','Corporate Author','Corporate Contributors','Former owner','Engraver']
pub_df.drop(droplist, inplace=True, axis=1)
print(pub_df.head())
print(pub_df.info())
print(pub_df.dtypes)
print(pub_df.shape)
print(pub_df.describe())   # for integer variables
'''print(pub_df.describe(include=[np.object])) # for object variables
# top=mode and freq= freq of mode value

print(pub_df.describe(include='all')) '''

place= pub_df['Place of Publication']
print(place)

lnd = place.str.contains('London')
print(lnd.head())
off=place.str.contains('Oxford')
print(off.head())

pub_df['Place of Publication'] = np.where(pub_df['Place of Publication'].str.contains('London'), 'London',pub_df['Place of Publication'])

print(pub_df['Place of Publication'])

pub_df.loc[667]
pub_df['Place of Publication'] = np.where(pub_df['Place of Publication'].str.contains('Oxford'), 'Oxford',pub_df['Place of Publication'])
pub_df.loc[667]
pub_df.loc[3017]
pub_df['Place of Publication']= pub_df['Place of Publication'].str.replace('-', ' ')
pub_df.loc[3017]

pub_df['Place of Publication'].is_unique

def unique(place):
 
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in place:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    for x in unique_list:
        print (x)
        
unique(place)

extr = pub_df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
extr.head()




