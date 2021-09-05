
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data = pd.read_excel(r'C:/Users/Administrator/Desktop/Time series/AirQualityUCI.xlsx', header = 0)


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[4]:


AQ1 = data.loc[:,["Date","Time","AH"]]


# In[5]:


AQ1.head()


# In[6]:


AQ1["DT"] = AQ1["Date"].astype(str) + " " + AQ1["Time"].astype(str)


# In[7]:


AQ1.drop(["Date", "Time"], axis = 1, inplace = True)


# In[10]:


AQ1["AH"].dtype


# In[8]:


AQ1.head()


# In[9]:


index = AQ1[AQ1["AH"]== -200].index[0]
print(index)


# In[10]:


AQ2 = AQ1.iloc[:index, :]


# In[14]:


AQ2.info()


# In[11]:


AQ2['DT'] = pd.to_datetime(AQ2['DT'])


# In[12]:


AQ2.info()


# In[13]:


AQ2 = AQ2.set_index("DT")


# In[14]:


AQ2.head()


# In[15]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(AQ2["AH"], model='additive')
fig = decomposition.plot()
print(plt.show())


# In[16]:


AQ2["AH"].plot(figsize=(15, 6))
plt.show()


# In[20]:


from statsmodels.tsa.ar_model import AR


# In[21]:


# fit model
model = AR(AQ2)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+1)
print(yhat)


# In[212]:


len(AQ2)


# In[22]:


# MA example
from statsmodels.tsa.arima_model import ARMA
from random import random
model = ARMA(AQ2, order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2))
print(yhat)


# In[218]:


# ARMA example
from statsmodels.tsa.arima_model import ARMA
# fit model
model = ARMA(AQ2, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2))
print(yhat)


# In[220]:


# ARIMA example
from statsmodels.tsa.arima_model import ARIMA
# fit model
model = ARIMA(AQ2, order=(1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2), typ='levels')
print(yhat)


# In[17]:


# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
# fit model
model = SARIMAX(AQ2, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+6)
print(yhat)


# In[20]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO


# In[233]:


friedman2 = requests.get('https://www.stata-press.com/data/r12/friedman2.dta').content
data = pd.read_stata(BytesIO(friedman2))
data.index = data.time


# In[235]:


data.head()


# In[236]:


endog = data.loc['1959':'1981', 'consump']
exog = sm.add_constant(data.loc['1959':'1981', 'm2'])


# In[239]:





# In[21]:


# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(AQ2.iloc[1:], lags=40, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(AQ2.iloc[1:], lags=40, ax=axes[1])


# In[ ]:


# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
# fit model
model = SARIMAX(AQ2, order=(1, 1, 30), seasonal_order=(1, 1, 1, 24))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+6)
print(yhat)


# In[26]:


from pandas import Series
from statsmodels.tsa.stattools import adfuller


# In[ ]:


data = pd.read_excel(r'C:/Users/Administrator/Desktop/Time series/AirQualityUCI.xlsx', header = 0)


# In[22]:


from pandas import Series
from statsmodels.tsa.stattools import adfuller
X = pd.Series(AQ2["AH"])
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[23]:


import itertools
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[26]:


type(param)


# In[24]:


import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
all_aic = []
all_param= []
#all_param_seasonl = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(AQ2,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            all_aic.append(results.aic)
            all_param.append([param, param_seasonal, results.aic])
            #print(results.aic)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
    


# In[25]:


min(all_aic)


# In[26]:


for j in range(0,len(all_param)):
    if min(all_aic)==all_param [j][2]:
        print(all_param[j])


# In[171]:


min(all_aic)

