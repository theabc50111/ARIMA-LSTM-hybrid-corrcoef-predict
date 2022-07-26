#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers.core import Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, LSTM, Activation
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ModelCheckpoint


# In[2]:


# Train - Dev - Test Generation
train_raw = pd.read_csv('./dataset/train.csv').drop('Unnamed: 0', axis=1)
dev_raw = pd.read_csv('./dataset/dev.csv').drop('Unnamed: 0', axis=1)
test1_raw = pd.read_csv('./dataset/test1.csv').drop('Unnamed: 0', axis=1)
test2_raw = pd.read_csv('./dataset/test2.csv').drop('Unnamed: 0', axis=1)

train_X = train_raw.iloc[:, :-1]
dev_X = dev_raw.iloc[:, :-1]
test1_X = test1_raw.iloc[:, :-1]
test2_X = test2_raw.iloc[:, :-1]
train_Y = train_raw.iloc[:, -1]
dev_Y = dev_raw.iloc[:, -1]
test1_Y = test1_raw.iloc[:, -1]
test2_Y = test2_raw.iloc[:, -1]


# data sampling
STEP = 20
#num_list = [STEP*i for i in range(int(1117500/STEP))]

_train_X = np.asarray(train_X).reshape((int(1117500/STEP), 20, 1))
_dev_X = np.asarray(dev_X).reshape((int(1117500/STEP), 20, 1))
_test1_X = np.asarray(test1_X).reshape((int(1117500/STEP), 20, 1))
_test2_X = np.asarray(test2_X).reshape((int(1117500/STEP), 20, 1))

_train_Y = np.asarray(train_Y).reshape(int(1117500/STEP), 1)
_dev_Y = np.asarray(dev_Y).reshape(int(1117500/STEP), 1)
_test1_Y = np.asarray(test1_Y).reshape(int(1117500/STEP), 1)
_test2_Y = np.asarray(test2_Y).reshape(int(1117500/STEP), 1)


# In[3]:


def double_tanh(x):
    return (K.tanh(x) * 2)

#get_custom_objects().update({'double_tanh':Activation(double_tanh)})

# Model Generation
model = Sequential()
#check https://machinelearningmastery.com/use-weight-regularization-lstm-networks-time-series-forecasting/
model.add(LSTM(25, input_shape=(20,1), dropout=0.0, kernel_regularizer=l1_l2(0.00,0.00), bias_regularizer=l1_l2(0.00,0.00)))
model.add(Dense(1, activation=double_tanh))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
#, kernel_regularizer=l1_l2(0,0.1), bias_regularizer=l1_l2(0,0.1),

model.summary()


# In[4]:


# Fitting the Model
model_scores = {}
Reg = False
d = 'LSTM_only_new_api'

if Reg :
    d += '_with_reg'

epoch_num=1
max_epoch = 3500
for _ in range(max_epoch):

    # train the model
    dir_ = './lstm_only_models/'+d
    file_list = os.listdir(dir_)
    if len(file_list) != 0 :
        epoch_num = len(file_list) + 1
        recent_model_name = 'epoch'+str(epoch_num-1)
        filepath = './lstm_only_models/' + d + '/' + recent_model_name
        # custom_objects = {"double_tanh": double_tanh}
        # with keras.utils.custom_object_scope(custom_objects):
        model = load_model(filepath)

    filepath = './lstm_only_models/' + d + '/epoch'+str(epoch_num)

    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
    model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True)
    model.save(filepath)
    
    #callbacks_list = [checkpoint]
    #if len(callbacks_list) == 0:
    #    model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True)
    #else:
    #    model.fit(_train_X, _train_Y, epochs=1, batch_size=500, shuffle=True, callbacks=callbacks_list)

    # test the model
    score_train = model.evaluate(_train_X, _train_Y)
    score_dev = model.evaluate(_dev_X, _dev_Y)
    score_test1 = model.evaluate(_test1_X, _test1_Y)
    score_test2 = model.evaluate(_test2_X, _test2_Y)

    print('train set score : mse - ' + str(score_train[1]) +' / mae - ' + str(score_train[2]))
    print('dev set score : mse - ' + str(score_dev[1]) +' / mae - ' + str(score_dev[2]))
    print('test1 set score : mse - ' + str(score_test1[1]) +' / mae - ' + str(score_test1[2]))
    print('test2 set score : mse - ' + str(score_test2[1]) +' / mae - ' + str(score_test2[2]))
#.history['mean_squared_error'][0]
    # get former score data
    df = pd.read_csv("./lstm_only_scores/"+d+".csv")
    train_mse = list(df['TRAIN_MSE'])
    dev_mse = list(df['DEV_MSE'])
    test1_mse = list(df['TEST1_MSE'])
    test2_mse = list(df['TEST2_MSE'])

    train_mae = list(df['TRAIN_MAE'])
    dev_mae = list(df['DEV_MAE'])
    test1_mae = list(df['TEST1_MAE'])
    test2_mae = list(df['TEST2_MAE'])

    # append new data
    train_mse.append(score_train[1])
    dev_mse.append(score_dev[1])
    test1_mse.append(score_test1[1])
    test2_mse.append(score_test2[1])

    train_mae.append(score_train[2])
    dev_mae.append(score_dev[2])
    test1_mae.append(score_test1[2])
    test2_mae.append(score_test2[2])

    # organize newly created score dataset
    model_scores['TRAIN_MSE'] = train_mse
    model_scores['DEV_MSE'] = dev_mse
    model_scores['TEST1_MSE'] = test1_mse
    model_scores['TEST2_MSE'] = test2_mse

    model_scores['TRAIN_MAE'] = train_mae
    model_scores['DEV_MAE'] = dev_mae
    model_scores['TEST1_MAE'] = test1_mae
    model_scores['TEST2_MAE'] = test2_mae
    
    # save newly created score dataset
    model_scores_df = pd.DataFrame(model_scores)
    model_scores_df.to_csv("./lstm_only_scores/"+d+".csv")


# In[ ]:

