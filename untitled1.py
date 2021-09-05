# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:07:35 2019

@author: suvanjeet
"""

#%%
import pandas as pd
import numpy as np
AQ=pd.read_excel(r'D:\Python day executions\files\AirQualityUCI.xlsx',header=0) 
#%%
AQ.head()
#%%
AQ1=AQ[['Date','Time','AH']]
AQ1.head()
#%%
AQ1['AH'].dtype
#%%
AQ1['DT']=AQ1['Date']+ ' '+ AQ1['Time']

#%%
#AQ1.set_index(['Date', 'Time'], drop=False)

AQ1['DT']=AQ1['Date'].astype(str)+ ' ' + AQ1['Time'].astype(str)
#%%
AQ1.head()

#%%
#print(AQ1.loc[AQ1['AH'] == -200])
#AQ1['AH'].where(AQ1['AH'] < -170)
#%%
index=AQ1[AQ1["AH"]== -200].index[0]
#print(index)

AQ2 = AQ1.iloc[:index, :]
#%%
AQ2.head()
#%%
AQ2= AQ2.drop(['Date','Time'], axis=1)
AQ2.head()
#%%
AQ2.tail()
#%%
AQ2.shape
#%%
AQ2['DT'] = pd.to_datetime(AQ2['DT'])
#%%
AQ2.head()
#%%

AQ2['DT'].dtype
#%%
AQ2 = AQ2.set_index('DT')
#%%
AQ2.head()
#%%
#pd.Series(AQ2)

#%%
import statsmodels.api as sm
#%%
from pylab import rcParams
rcParams['figure.figsize'] = 8, 4
decomposition = sm.tsa.seasonal_decompose(AQ2, model='additive')
fig = decomposition.plot()
plt.show()
#%%
AQ2.plot(figsize=(15, 6))
plt.show()

#%%
from statsmodels.tsa.ar_model import AR
model = AR(AQ2)
model_fit = model.fit()
yhat = model_fit.predict(len(AQ2), len(AQ2)+6)
print(yhat)

#%%
from statsmodels.tsa.arima_model import ARMA
model = ARMA(AQ2, order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+6)
print(yhat)

#%%
model = ARMA(AQ2, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+6)
print(yhat)
#%%
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(AQ2, order=(1, 1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+6, typ='levels')
print(yhat)

#%%
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(AQ2, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+6)
print(yhat)
#%%
# contrived dataset
data1 = [x + random() for x in range(1, 100)]
data2 = [x + random() for x in range(101, 200)]
# fit model
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
exog2 = [200 + random()]
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
print(yhat)

#%%
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
#%%
# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(AQ2.iloc[1:], lags=40, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(AQ2.iloc[1:], lags=40, ax=axes[1])

#%%
from statsmodels.tsa.stattools import adfuller
#%%
AQ2.head()

#%%
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    rolmean=AQ2['AH'].rolling(window=12).mean()
    rolstd=AQ2['AH'].rolling(window=12).mean()
    #Determing rolling statistics
    #rolmean = pd.rolling_mean(timeseries, window=12)
    #rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
#%%
ts = AQ2['AH']
test_stationarity(ts)

#%%
# Graph data
fig, axes = plt.subplots(1, 2, figsize=(15,4))

fig = sm.graphics.tsa.plot_acf(AQ2.iloc[1:], lags=40, ax=axes[0])
fig = sm.graphics.tsa.plot_pacf(AQ2.iloc[1:], lags=40, ax=axes[1])

#%%
#%%
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(AQ2, order=(3 ,1 , 30), seasonal_order=(1, 1, 1, 24))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(AQ2), len(AQ2)+6)
print(yhat)

#%%
import itertools
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#%%
import warnings
warnings.filterwarnings("ignore")
all_aic = []
all_param= []
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
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#%%
min(all_aic)

#%%
for j in range(0,len(all_param)):
    if min(all_aic)==all_param [j][2]:
        print(all_param[j])
        
#%%


















