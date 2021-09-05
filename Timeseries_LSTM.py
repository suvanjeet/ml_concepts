import keras
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 

AQ=pd.read_excel(r'D:\Python day executions\files\AirQualityUCI.xlsx',header=0) 


AQ1=AQ[['Date','Time','AH']]

AQ1['DT']=AQ1['Date'].astype(str)+ ' ' + AQ1['Time'].astype(str)
AQ1.head()

outlier_index=AQ1[AQ1["AH"]== -200].index[0]

AQ2 = AQ1.iloc[:outlier_index, :]
AQ2= AQ2.drop(['Date','Time'], axis=1)
AQ2.tail()

AQ2['DT'] = pd.to_datetime(AQ2['DT'])

AQ2 = AQ2.set_index('DT')
AQ2.head()

training_processed = AQ2.iloc[:400, :]
print(training_processed)

from sklearn.preprocessing import MinMaxScaler  
scaler = MinMaxScaler(feature_range = (0, 1))

training_scaled = scaler.fit_transform(training_processed)  

features_set = []  
labels = []  
for i in range(1, 301):  
    features_set.append(training_scaled[i-1:i+99, 0])
    labels.append(training_scaled[i+99, 0])
    
features_set, labels = np.array(features_set), np.array(labels) 
features_set

features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1)) 

from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.layers import Dropout 

model = Sequential()  
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1],1)))

model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))

model.add(LSTM(units=50))  
model.add(Dropout(0.2)) 


model.add(Dense(units = 1))  

model.compile(optimizer = 'adam', loss = 'mean_squared_error') 

model.fit(features_set, labels, epochs = 100, batch_size = 32) 

testing = AQ2.iloc[302:523, :] 

testing_processed= scaler.transform(testing)
  
features_set_test = []  
labels_test = []  
for j in range(1, 123):  
    features_set_test.append(testing_processed[j-1:j+99, 0])
    labels_test.append(testing_processed[j+99, 0])
    
features_set_test = np.array(features_set_test)
features_set_test = np.reshape(features_set_test, (features_set_test.shape[0], features_set_test.shape[1], 1)) 


yhat = model.predict(features_set_test, verbose=0)

predictions = scaler.inverse_transform(yhat)  
predictions

testing.head()
test= testing.reset_index(drop=True)
test1= test[ -122:]
test1= test1.reset_index(drop=True)
test1

plt.figure(figsize=(10,6))  
plt.plot(test1, color='blue', label='Actual AH')  
plt.plot(predictions , color='red', label='Predicted AH')  
plt.title('AH Prediction')  
plt.xlabel('Date')  
plt.ylabel('AH Value')  
plt.legend()  
plt.show() 
