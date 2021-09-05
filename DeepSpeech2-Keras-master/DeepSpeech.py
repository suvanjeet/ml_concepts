
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.metrics import categorical_accuracy, mean_squared_error
from tensorflow.python.keras.callbacks import BaseLogger, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Ones, Zeros, glorot_normal
from tensorflow.python.framework import tensor_shape
from data_generator import DataGenerator

import numpy as np


# In[2]:


def clipped_relu(x):
    return relu(x, max_value=20)

def ctc_lambda_func(args):
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc(y_true, y_pred):
    return y_pred


# In[23]:


def get_speech_model():
    model = Sequential()
    
    # Batch normalize the input
    model.add(BatchNormalization(axis=-1, input_shape=(None, 161), name='BN_1'))
    
    # 1D Convs
    model.add(Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_1'))
    model.add(Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_2'))
    model.add(Conv1D(512, 5, strides=2, activation=clipped_relu, name='Conv1D_3'))
    
    # Batch Normalization
    model.add(BatchNormalization(axis=-1, name='BN_2'))
    
    # BiRNNs
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_1'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_2'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_3'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_4'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_5'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_6'), merge_mode='sum'))
    model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_7'), merge_mode='sum'))
    
    # Batch Normalization
    model.add(BatchNormalization(axis=-1, name='BN_3'))
    
    # FC
    model.add(TimeDistributed(Dense(1024, activation=clipped_relu, name='FC1')))
    model.add(TimeDistributed(Dense(29, activation='softmax', name='y_pred')))
    return model

def get_trainable_speech_model():
    model = get_speech_model()
    y_pred = model.outputs[0]
    model_input = model.inputs[0]
    
    model.summary()
    
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    loss_out = Lambda(ctc_lambda_func, name='ctc')([labels, y_pred, input_length, label_length])
    trainable_model = Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
    return trainable_model


# In[24]:


model = get_trainable_speech_model()
model.summary()


# In[23]:


data_gen = DataGenerator()
data_gen.load_test_data('data/LibriSpeechTest/')
data_gen.load_validation_data('data/LibriSpeechDev/')
data_gen.load_train_data('data/LibriSpeechTrain/')
data_gen.fit_train(100)


# In[24]:


assert(len(data_gen.train_audio_paths) == len(data_gen.train_durations)) 
assert(len(data_gen.train_durations) == len(data_gen.train_texts))

assert(len(data_gen.val_audio_paths) == len(data_gen.val_durations)) 
assert(len(data_gen.val_durations) == len(data_gen.val_texts))

assert(len(data_gen.test_audio_paths) == len(data_gen.test_durations)) 
assert(len(data_gen.test_durations) == len(data_gen.test_texts))

print('Train set:', len(data_gen.train_audio_paths), 
      '\nVal set:', len(data_gen.val_audio_paths), 
      '\nTest set:', len(data_gen.test_audio_paths))

