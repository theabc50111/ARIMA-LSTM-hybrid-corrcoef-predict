#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
from pmdarima.arima import ARIMA, auto_arima
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
import warnings
import logging
warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
err_log_handler = logging.FileHandler(filename="./models/arima_train_err_log.txt", mode='a')        
err_logger = logging.getLogger("arima_train_err")
err_logger.addHandler(err_log_handler)

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
logging.info(f"len(train_set): {len(portfolio_train)}, len(all_set): {len(portfolio_all)}, len(test_set): {len(portfolio_other)}")
# evaluation set
train_info = {"paper_eva_1": {"portfolio": ['PRGO', 'MRO', 'ADP', 'HCP', 'FITB', 'PEG', 'SYMC', 'EOG', 'MDT', 'NI'], "file_name": "paper_eva_1_res"},
            "paper_eva_2": {"portfolio": ['STI', 'COP', 'MCD', 'AON', 'JBHT', 'DISH', 'GS', 'LRCX', 'CTXS', 'LEG'], "file_name": "paper_eva_2_res"},
            "paper_eva_3": {"portfolio": ['TJX', 'EMN', 'JCI', 'C', 'BIIB', 'HOG', 'PX', 'PH', 'XEC', 'JEC'], "file_name": "paper_eva_3_res"},
            "paper_eva_4": {"portfolio": ['ROP', 'AZO', 'URI', 'TROW', 'CMCSA', 'SLB', 'VZ', 'MAC', 'ADS', 'MCK'], "file_name": "paper_eva_4_res"},
            "paper_eva_5": {"portfolio": ['RL', 'CVX', 'SRE', 'PFE', 'PCG', 'UTX', 'NTRS', 'INCY', 'COP', 'HRL'], "file_name": "paper_eva_5_res"},
            "445_all": {"portfolio": portfolio_all, "file_name": "sp500_20082017_all_res"},
            "150_train": {"portfolio": portfolio_train, "file_name": "sp500_20082017_train_res"},
            "295_test": {"portfolio": portfolio_other, "file_name": "sp500_20082017_test_res"},
           }


data_implement = "150_train"
portfolio_implement = train_info[data_implement]['portfolio']
output_file_name = train_info[data_implement]['file_name']
fig_title = data_implement

# setting of output files
save_raw_corr_data = True
save_train_info_arima_resid_data = True


pd.to_datetime(stock_price_df['Date'], format='%Y-%m-%d')
stock_price_df = stock_price_df.set_index(pd.DatetimeIndex(stock_price_df['Date']))


# In[ ]:


def gen_data_corr(portfolio: list, corr_ind: list) -> "pd.DataFrame":
    tmp_corr = stock_price_df[portfolio[0]].rolling(window=100).corr(stock_price_df[portfolio[1]])
    tmp_corr = tmp_corr.iloc[corr_ind].values
    data_df = pd.DataFrame(tmp_corr.reshape(-1, 24), dtype="float32")
    ind = [f"{portfolio[0]} & {portfolio[1]}_{i}" for i in range(0, 100, 20)]
    data_df.index = ind
    return data_df


def gen_train_data(portfolio: list, corr_ind: list, save_file: bool = False)-> "four pd.DataFrame":
    train_df = pd.DataFrame(dtype="float32")
    dev_df = pd.DataFrame(dtype="float32")
    test1_df = pd.DataFrame(dtype="float32")
    test2_df = pd.DataFrame(dtype="float32")

    for pair in tqdm(combinations(portfolio, 2)):
        data_df = gen_data_corr([pair[0], pair[1]], corr_ind=corr_ind)
        data_split = {'train': [0, 21], 'dev': [1, 22], 'test1': [2, 23], 'test2': [3, 24]}
        train_df = pd.concat([train_df, data_df.iloc[:, 0:21]])
        dev_df = pd.concat([dev_df, data_df.iloc[:, 1:22]])
        test1_df = pd.concat([test1_df, data_df.iloc[:, 2:23]])
        test2_df = pd.concat([test2_df, data_df.iloc[:, 3:24]])

    if save_file:
        Path('./dataset/before_arima/').mkdir(parents=True, exist_ok=True)
        train_df.to_csv(f"./dataset/before_arima/{output_file_name}_train.csv")
        dev_df.to_csv(f"./dataset/before_arima/{output_file_name}_dev.csv")
        test1_df.to_csv(f"./dataset/before_arima/{output_file_name}_test1.csv")
        test2_df.to_csv(f"./dataset/before_arima/{output_file_name}_test2.csv")

    return train_df, dev_df, test1_df, test2_df 


corr_ind = list(range(99, 2400, 100)) + list(range(99+20, 2500, 100)) + list(range(99+40, 2500, 100)) + list(range(99+60, 2500, 100)) + list(range(99+80, 2500, 100))
corr_datasets = gen_train_data(portfolio_implement, corr_ind, save_file = save_raw_corr_data)


# # ARIMA model

# In[ ]:


def arima_model(dataset: "pd.DataFrame", save_file_period: str = "") -> ("pd.DataFrame", "pd.DataFrame", "pd.DataFrame"):
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
    arima_output_list = []
    arima_resid_list = []
    arima_model_info_list = []
    for corr_pair, corr_series in dataset.iterrows():
        while not find_arima_model:
            try:
                for model_key in model_dict:
                    if model_key not in tested_models:
                        test_model = model_dict[model_key].fit(corr_series[:-1]) # only use first 20 corrletaion coefficient to fit ARIMA model
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
                    err_logger.error(f"fatal error, {corr_pair} doesn't have appropriate arima model\n", exc_info=True)
                    break
            else:
                #model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210, "model_330": model_330}
                model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210}
                tested_models.clear()
                find_arima_model = True
        try:
            arima_pred = list(arima_model.predict(n_periods=1))
        except Exception:
            err_logger.error(f"{corr_pair} in {save_file_period} be predicted by {arima_model_name}(its aic:{arima_model.aic()}) getting error:\n", exc_info=True)
            dataset = dataset.drop(index=corr_pair)
        else:
            arima_pred_in_sample = list(arima_model.predict_in_sample())
            arima_pred_in_sample = [np.mean(arima_pred_in_sample[1:])] + arima_pred_in_sample[1:]
            arima_output = arima_pred_in_sample + arima_pred
            arima_output = np.clip(np.array(arima_output), -1, 1)
            arima_output_list.append(arima_output)
            
            arima_resid = pd.Series(np.array(corr_series) - arima_output)
            arima_resid_list.append(np.array(arima_resid))

            arima_info = [corr_pair, arima_model_name]
            for attr in ["aic", "pvalues", "params", "arparams", "aroots", "maparams", "maroots"]:
                try:
                    val = getattr(arima_model, attr)()
                except AttributeError:
                    arima_info.append(None)
                else:
                    arima_info.append(val)
            else:
                arima_model_info_list.append(arima_info) 
        finally:
            find_arima_model = False


    arima_model_info_df = pd.DataFrame(arima_model_info_list, dtype="float32", columns=["items", "model_name", "aic", "pvalues", "params", "arparams", "aroots", "maparams", "maroots"]).set_index("items")
    arima_output_df = pd.DataFrame(arima_output_list, dtype="float32", index=dataset.index)
    arima_resid_df = pd.DataFrame(arima_resid_list, dtype="float32", index=dataset.index)

    if save_file_period:
        Path('./dataset/after_arima').mkdir(parents=True, exist_ok=True)
        arima_model_info_df.to_csv(f'./dataset/after_arima/{output_file_name}_arima_model_info_{save_file_period}.csv')
        arima_output_df.to_csv(f'./dataset/after_arima/{output_file_name}_arima_output_{save_file_period}.csv')
        arima_resid_df.to_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_{save_file_period}.csv')

    return arima_model_info_df, arima_output_df, arima_resid_df


#for (file_name, dataset) in tqdm(zip(['train', 'dev', 'test1', 'test2'], corr_datasets)):
#    if save_train_info_arima_resid_data:
#        arima_model(dataset, save_file_period=file_name)
#    else:
#        arima_model(dataset)


# # LSTM


# Dataset.from_tensor_slices(dict(pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv')))
lstm_train_X = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_train.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_train_Y = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_train.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_dev_X = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_dev.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_dev_Y = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_dev.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test1_X = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_test1.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test1_Y = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_test1.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test2_X = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_test2.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test2_Y = pd.read_csv(f'./dataset/after_arima/{output_file_name}_arima_resid_test2.csv').set_index('Unnamed: 0').iloc[::, -1]


lstm_train_X = lstm_train_X.values.reshape(-1, 20, 1)
lstm_train_Y = lstm_train_Y.values.reshape(-1, 1)
lstm_dev_X = lstm_dev_X.values.reshape(-1, 20, 1)
lstm_dev_Y = lstm_dev_Y.values.reshape(-1, 1)
lstm_test1_X = lstm_test1_X.values.reshape(-1, 20, 1)
lstm_test1_Y = lstm_test1_Y.values.reshape(-1, 1)
lstm_test2_X = lstm_test2_X.values.reshape(-1, 20, 1)
lstm_test2_Y = lstm_test2_Y.values.reshape(-1, 1)


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


model_dir = './models'
log_dir = './models/lstm_train_logs'
res_dir = './results'
os.makedirs(res_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
res_csv_path = Path(res_dir+'/LSTM_evaluation.csv')
res_csv_path.touch(exist_ok=True)
with open(res_csv_path, 'r+') as f:
    if not f.read():
        f.write("epoch,TRAIN_MSE,DEV_MSE,TEST1_MSE,TEST2_MSE,TRAIN_MAE,DEV_MAE,TEST1_MAE,TEST2_MAE")

res_df = pd.read_csv(res_csv_path)
saved_model_list = [int(p.stem.split('_')[1]) for p in Path(model_dir).glob('*.h5')]
model_cbk = TensorBoard(log_dir=log_dir)
epoch_start = max(saved_model_list) if saved_model_list else 1
max_epoch = 3000

for epoch_num in range(epoch_start, max_epoch):
    if epoch_num > 1:
        lstm_model = load_model(Path(model_dir)/f"{output_file_name}_epoch_{epoch_num - 1}.h5", custom_objects={'double_tanh':double_tanh})

    save_model = ModelCheckpoint(Path(model_dir)/f"{output_file_name}_epoch_{epoch_num}.h5",
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

