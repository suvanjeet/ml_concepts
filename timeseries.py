# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 23:08:22 2018

@author: suvanjeet
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
#%%

ap=pd.read_csv(r'D:\Python day executions\files\AirPassengers.csv',header=0) 
#%%
ap.head()

#%%
ap.dtypes

#%%
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv(r'D:\Python day executions\files\AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
data.head()

#%%
ts = data['#Passengers'] 
ts.head(10)

#%%
plt.plot(ts)

#%%
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    rolmean=data['#Passengers'].rolling(window=12).mean()
    rolstd=data['#Passengers'].rolling(window=12).mean()
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
test_stationarity(ts)

#%%

ts_log = np.log(ts)
plt.plot(ts_log) 

#%%
moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

#%%
ts_log_moving_avg_diff = ts_log - moving_avg
#%%
ts_log_moving_avg_diff.head(22)

#%%
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

#%%
plt.plot(ts_log_moving_avg_diff)
#%%
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_moving_avg_diff, nlags=20)
lag_pacf = pacf(ts_log_moving_avg_diff, nlags=20, method='ols')

#%%
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#%%
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_moving_avg_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#%%
from statsmodels.tsa.arima_model import ARIMA

#%%
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_moving_avg_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_moving_avg_diff)**2))

#%%
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.head()

#%%
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

#%%
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

#%%
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
#%%
data['#Passengers'].max


#end--------------------------------------------------------------------------------------------


#%%

     
df=pd.read_excel(r'D:\Python day executions\files\Sample - Superstore.xls',header=0)
df.head() 

#%%
furniture = df.loc[df['Category'] == 'Furniture']
furniture.head()

#%%
furniture.info()
#%%
furniture['Order Date'].min(), furniture['Order Date'].max()
#%%
df.columns
#%%
cols= ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
       'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Quantity', 'Discount', 'Profit']

#%%
furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
furniture.isnull().sum()

#%%
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
furniture.head()

#%%

furniture.shape
#%%

furniture = furniture.set_index('Order Date')
furniture.index

#%%
y = furniture['Sales'].resample('MS').mean()
#%%
y['2017':]

#%%
y.plot(figsize=(8, 4))
plt.show()

#%%
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

#%%
from pylab import rcParams
rcParams['figure.figsize'] = 8, 4
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#%%
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(y, order=(1,1,1))
results = model.fit()
print(results.summary())

#%%

p=[0,1]
d=[0,1]
q=[0,1]
for i in p:
    for j in d:
        for k in q:
            model = ARIMA(y, order=(i,j,k))
            results = model.fit()
            print(results.aic)
            
#%%

results.plot_diagnostics(figsize=(16, 8))
plt.show()

#%%

pred = results.predict(start=pd.to_datetime('2017-01-01'), dynamic=False) 
pred           


















