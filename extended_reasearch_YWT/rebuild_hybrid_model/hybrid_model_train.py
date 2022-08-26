#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import os
import pathlib
import random
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima.arima import ARIMA, auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats as st
from sklearn.linear_model import LinearRegression
# from keras.models import Sequential, load_model
# from keras.layers import Dense, LSTM, Activation
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
# from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.data import Dataset
import warnings
warnings.simplefilter("ignore")

# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501


# # Prepare data

# In[2]:


stock_price_df = pd.read_csv("../../stock08_price.csv")

universe = list(stock_price_df.columns.values[1:])
universe.remove("SP500")
# train data
portfolio_train = ['CELG', 'PXD', 'WAT', 'LH', 'AMGN', 'AOS', 'EFX', 'CRM', 'NEM', 'JNPR', 'LB', 'CTAS', 'MAT', 'MDLZ', 'VLO', 'APH', 'ADM', 'MLM', 'BK', 'NOV', 'BDX', 'RRC', 'IVZ', 'ED', 'SBUX', 'GRMN', 'CI', 'ZION', 'COO', 'TIF', 'RHT', 'FDX', 'LLL', 'GLW', 'GPN', 'IPGP', 'GPC', 'HPQ', 'ADI', 'AMG', 'MTB', 'YUM', 'SYK', 'KMX', 'AME', 'AAP', 'DAL', 'A', 'MON', 'BRK', 'BMY', 'KMB', 'JPM', 'CCI', 'AET', 'DLTR', 'MGM', 'FL', 'HD', 'CLX', 'OKE', 'UPS', 'WMB', 'IFF', 'CMS', 'ARNC', 'VIAB', 'MMC', 'REG', 'ES', 'ITW', 'NDAQ', 'AIZ', 'VRTX', 'CTL', 'QCOM', 'MSI', 'NKTR', 'AMAT', 'BWA', 'ESRX', 'TXT', 'EXR', 'VNO', 'BBT', 'WDC', 'UAL', 'PVH', 'NOC', 'PCAR', 'NSC', 'UAA', 'FFIV', 'PHM', 'LUV', 'HUM', 'SPG', 'SJM', 'ABT', 'CMG', 'ALK', 'ULTA', 'TMK', 'TAP', 'SCG', 'CAT', 'TMO', 'AES', 'MRK', 'RMD', 'MKC', 'WU', 'ACN', 'HIG', 'TEL', 'DE', 'ATVI', 'O', 'UNM', 'VMC', 'ETFC', 'CMA', 'NRG', 'RHI', 'RE', 'FMC', 'MU', 'CB', 'LNT', 'GE', 'CBS', 'ALGN', 'SNA', 'LLY', 'LEN', 'MAA', 'OMC', 'F', 'APA', 'CDNS', 'SLG', 'HP', 'XLNX', 'SHW', 'AFL', 'STT', 'PAYX', 'AIG', 'FOX', 'MA']
# all data
portfolio_all = universe
# all data - train data
portfolio_other = [p for p in universe if p not in portfolio_train]
print(len(portfolio_train), len(portfolio_all), len(portfolio_other))

#paper_eva_info = {"paper_eva_1": {"portfolio": ['PRGO', 'MRO', 'ADP', 'HCP', 'FITB', 'PEG', 'SYMC', 'EOG', 'MDT', 'NI'], "file_name": "paper_eva_1_res"}, 
#                  "paper_eva_2": {"portfolio": ['STI', 'COP', 'MCD', 'AON', 'JBHT', 'DISH', 'GS', 'LRCX', 'CTXS', 'LEG'], "file_name": "paper_eva_2_res"},
#                  "paper_eva_3": {"portfolio": ['TJX', 'EMN', 'JCI', 'C', 'BIIB', 'HOG', 'PX', 'PH', 'XEC', 'JEC'], "file_name": "paper_eva_3_res"},
#                  "paper_eva_4": {"portfolio": ['ROP', 'AZO', 'URI', 'TROW', 'CMCSA', 'SLB', 'VZ', 'MAC', 'ADS', 'MCK'], "file_name": "paper_eva_4_res"},
#                  "paper_eva_5": {"portfolio": ['RL', 'CVX', 'SRE', 'PFE', 'PCG', 'UTX', 'NTRS', 'INCY', 'COP', 'HRL'], "file_name": "paper_eva_5_res"},}
#
#paper_eva_implement = "paper_eva_5"
#portfolio_implement = paper_eva_info[paper_eva_implement]['portfolio']
#output_file_name = paper_eva_info[paper_eva_implement]['file_name']
#fig_title = paper_eva_implement

paper_eva_implement = "150_train"
portfolio_implement = portfolio_train
output_file_name = "150_train_res"
fig_title = paper_eva_implement


pd.to_datetime(stock_price_df['Date'], format='%Y-%m-%d')
stock_price_df = stock_price_df.set_index(pd.DatetimeIndex(stock_price_df['Date']))
# display(stock_price_df)


# In[6]:


def gen_data_corr(portfolio: list, corr_ind: list) -> "pd.DataFrame":
    tmp_corr = stock_price_df[portfolio[0]].rolling(window=100).corr(stock_price_df[portfolio[1]])
    tmp_corr = tmp_corr.iloc[corr_ind].values
    data_df = pd.DataFrame(tmp_corr.reshape(-1, 24))
    ind = [f"{portfolio[0]} & {portfolio[1]}_{i}" for i in range(0, 100, 20)]
    data_df.index = ind
    return data_df


def gen_train_data(portfolio: list, corr_ind: list, save_file: bool = False)-> "four pd.DataFrame":
    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test1_df = pd.DataFrame()
    test2_df = pd.DataFrame()

    for pair in tqdm(combinations(portfolio, 2)):
        data_df = gen_data_corr([pair[0], pair[1]], corr_ind=corr_ind)
        data_split = {'train': [0, 21], 'dev': [1, 22], 'test1': [2, 23], 'test2': [3, 24]}
        train_df = pd.concat([train_df, data_df.iloc[:, 0:21]])
        dev_df = pd.concat([dev_df, data_df.iloc[:, 1:22]])
        test1_df = pd.concat([test1_df, data_df.iloc[:, 2:23]])
        test2_df = pd.concat([test2_df, data_df.iloc[:, 3:24]])

    if save_file:
        pathlib.Path('./dataset/before_arima/').mkdir(parents=True, exist_ok=True)
        train_df.to_csv("./dataset/before_arima/train.csv")
        dev_df.to_csv("./dataset/before_arima/dev.csv")
        test1_df.to_csv("./dataset/before_arima/test1.csv")
        test2_df.to_csv("./dataset/before_arima/test2.csv")

    return train_df, dev_df, test1_df, test2_df 


corr_ind = list(range(99, 2400, 100)) + list(range(99+20, 2500, 100)) + list(range(99+40, 2500, 100)) + list(range(99+60, 2500, 100)) + list(range(99+80, 2500, 100))
corr_datasets = gen_train_data(portfolio_implement, corr_ind, save_file = True)


# # ARIMA model

# In[7]:


def arima_model(dataset: "pd.DataFrame", save_file_name: str = "") -> ("pd.DataFrame", "pd.DataFrame", "pd.DataFrame"):
    model_110 = ARIMA(order=(1, 1, 0), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_011 = ARIMA(order=(0, 1, 1), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_111 = ARIMA(order=(1, 1, 1), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_211 = ARIMA(order=(2, 1, 1), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    model_210 = ARIMA(order=(2, 1, 0), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)
    #model_330 = ARIMA(order=(3, 3, 0), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)

    #model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210, "model_330": model_330}
    model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210}
    tested_models = []
    arima_model = None
    find_arima_model = False
    arima_model_name_list = []
    arima_pred_list = []
    residual = []
    for s in dataset.iterrows():
        while not find_arima_model:
            try:
                for model_key in model_dict:
                    if model_key not in tested_models:
                        test_model = model_dict[model_key].fit(s[1])
                        if arima_model is None:
                            arima_model = test_model
                            arima_model_name = model_key
                        elif arima_model.aic() <= test_model.aic():
                            pass
                        else:
                            arima_model = test_model
                            arima_model_name = model_key
                    tested_models.append(model_key)

            except Exception:
                if len(model_dict)-1 != 0:
                    del model_dict[model_key]
                else:
                    print(f"fatal error, {s[1].name} doesn't have appropriate arima model")
                    break
            else:
                #model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210, "model_330": model_330}
                model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210}
                tested_models.clear()
                find_arima_model = True

        arima_pred = list(arima_model.predict_in_sample())
        arima_pred = [np.mean(arima_pred[1:])] + arima_pred[1:]
        arima_pred = np.clip(np.array(arima_pred), -1, 1)

        res = pd.Series(np.array(s[1]) - arima_pred)
        arima_model_name_list.append(arima_model_name)
        arima_pred_list.append(arima_pred)
        residual.append(np.array(res))
        find_arima_model = False
    arima_model_name_df = pd.DataFrame(arima_model_name_list)
    arima_pred_df = pd.DataFrame(arima_pred_list)
    arima_resid_df = pd.DataFrame(residual)
    arima_model_name_df.index = dataset.index
    arima_pred_df.index = dataset.index
    arima_resid_df.index = dataset.index

    if save_file_name:
        pathlib.Path('./dataset/after_arima').mkdir(parents=True, exist_ok=True)
        arima_model_name_df.to_csv(f'./dataset/after_arima/arima_model_{save_file_name}.csv')
        arima_pred_df.to_csv(f'./dataset/after_arima/arima_pred_{save_file_name}.csv')
        arima_resid_df.to_csv(f'./dataset/after_arima/arima_resid_{save_file_name}.csv')

    return arima_model_name_df, arima_pred_df, arima_resid_df


# In[8]:


for (file_name, dataset) in tqdm(zip(['train', 'dev', 'test1', 'test2'], corr_datasets)):
    arima_model(dataset, save_file_name=file_name)


# # LSTM

# In[9]:


# Dataset.from_tensor_slices(dict(pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv')))
lstm_train_X = pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_train_Y = pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_dev_X = pd.read_csv(f'./dataset/after_arima/arima_resid_dev.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_dev_Y = pd.read_csv(f'./dataset/after_arima/arima_resid_dev.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test1_X = pd.read_csv(f'./dataset/after_arima/arima_resid_test1.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test1_Y = pd.read_csv(f'./dataset/after_arima/arima_resid_test1.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test2_X = pd.read_csv(f'./dataset/after_arima/arima_resid_test2.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test2_Y = pd.read_csv(f'./dataset/after_arima/arima_resid_test2.csv').set_index('Unnamed: 0').iloc[::, -1]
num_samples = len(lstm_train_X)

lstm_train_X = lstm_train_X.values.reshape(num_samples, 20, 1)
lstm_train_Y = lstm_train_Y.values.reshape(num_samples, 1)
lstm_dev_X = lstm_dev_X.values.reshape(num_samples, 20, 1)
lstm_dev_Y = lstm_dev_Y.values.reshape(num_samples, 1)
lstm_test1_X = lstm_test1_X.values.reshape(num_samples, 20, 1)
lstm_test1_Y = lstm_test1_Y.values.reshape(num_samples, 1)
lstm_test2_X = lstm_test2_X.values.reshape(num_samples, 20, 1)
lstm_test2_Y = lstm_test2_Y.values.reshape(num_samples, 1)


# In[10]:


def double_tanh(x):
    return (tf.math.tanh(x) *2)


def build_many_one_lstm():
    inputs = Input(shape=(20, 1))
    lstm_1 = LSTM(units=25, kernel_regularizer=l1_l2(0.0, 0.0), bias_regularizer=l1_l2(0.0, 0.0))(inputs)
    outputs = Dense(units=1, activation=double_tanh)(lstm_1)
    return keras.Model(inputs, outputs, name="many_one_lstm")


lstm_model = build_many_one_lstm()
lstm_model.summary()
lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])


# In[11]:


model_dir = './models'
log_dir = './models/lstm_train_logs'
res_dir = './results'
os.makedirs(res_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
res_csv_path = pathlib.Path(res_dir+'/LSTM_evaluation.csv')
res_csv_path.touch(exist_ok=True)
with open(res_csv_path, 'r+') as f:
    if not f.read():
        f.write("epoch,TRAIN_MSE,DEV_MSE,TEST1_MSE,TEST2_MSE,TRAIN_MAE,DEV_MAE,TEST1_MAE,TEST2_MAE")

res_df = pd.read_csv(res_csv_path)
saved_model_list = [int(p.stem.split('_')[1]) for p in pathlib.Path(model_dir).glob('*.h5')]
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
epoch_start = max(saved_model_list) if saved_model_list else 1
max_epoch = 3000

for epoch_num in range(epoch_start, max_epoch):
    if epoch_num > 1:
        lstm_model = load_model(pathlib.Path(model_dir)/f"epoch_{epoch_num - 1}.h5", custom_objects={'double_tanh':double_tanh})

    save_model = keras.callbacks.ModelCheckpoint(pathlib.Path(model_dir)/f"epoch_{epoch_num}.h5",
                                                 monitor='loss', verbose=1, mode='min', save_best_only=False)
    lstm_model.fit(lstm_train_X, lstm_train_Y, epochs=1, batch_size=500, shuffle=True, callbacks=[model_cbk, save_model])
    
    # test the model
    score_train = lstm_model.evaluate(lstm_train_X, lstm_train_Y)
    score_dev = lstm_model.evaluate(lstm_dev_X, lstm_dev_Y)
    score_test1 = lstm_model.evaluate(lstm_test1_X, lstm_test1_Y)
    score_test2 = lstm_model.evaluate(lstm_test2_X, lstm_test2_Y)
    res_each_epoch_df = pd.DataFrame(np.array([epoch_num, score_train[0], score_dev[0], 
                                               score_test1[0], score_test2[0], 
                                               score_train[1], score_dev[1], 
                                               score_test1[1], score_test2[1]]).reshape(-1, 9),
                                    columns=["epoch", "TRAIN_MSE", "DEV_MSE", "TEST1_MSE", 
                                             "TEST2_MSE", "TRAIN_MAE", "DEV_MAE",
                                             "TEST1_MAE","TEST2_MAE"])
    res_df = pd.concat([res_df, res_each_epoch_df])

res_df.to_csv(res_csv_path, index=False)



