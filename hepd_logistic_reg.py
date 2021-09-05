# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 16:20:32 2018

@author: suvanjeet
"""

hepd = pd.read_csv(r'C:\Users\Administrator\Desktop\Data Science Program\Python\my data\hepatitis.csv', header = None, index_col = None)

hepd.head()
#%%
hepd.columns = ['Class', 'AGE', 'SEX', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK_PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME','HISTOLOGY']

#%%
hepd.head()
#%%
hepd.info()

#%%
hepd.replace(['?'], np.nan, inplace = True)

#%%
hepd.isnull().sum()

#%%
hepd.describe(include = 'all')

#%%
m_o = ['STEROID', 'FATIGUE', 'MALAISE', 'ANOREXIA', 'LIVER_BIG', 'LIVER_FIRM', 'SPLEEN_PALPABLE' ,'SPIDERS', 'ASCITES', 'VARICES', 'BILIRUBIN', 'ALK_PHOSPHATE', 'SGOT', 'ALBUMIN' ,'PROTIME']

#%%
for x in m_o:
hepd.fillna(hepd[x].mode()[0], inplace = True)
#%%
hepd.isnull().sum()

#%%
hepd.STEROID.astype(int)
hepd.FATIGUE.astype(int)
#%%

hepd.MALAISE.astype(int)
#%%

hepd.ANOREXIA.astype(int)
#%%

hepd.LIVER_BIG.astype(int)
#%%

hepd.LIVER_FIRM.astype(int)
#%%

hepd.SPLEEN_PALPABLE.astype(int)
#%%
hepd.SPIDERS.astype(int)
#%%
hepd.ASCITES.astype(int)
#%%

hepd.VARICES.astype(int)
#%%
hepd.ALK_PHOSPHATE.astype(int)
#%%

hepd.SGOT.astype(int)
#%%

hepd.PROTIME.astype(int)

#%%

#%%
# 2 = live
# 1 = die
#%%
x = hepd.values[:,1:]
y = hepd.values[:,0]

#%%

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, random_state = 10)

#%%
lrmodel = LogisticRegression()
lrmodel.fit(train_x, train_y)

#%%

y_pred= lrmodel.predict(test_x)


#%%
