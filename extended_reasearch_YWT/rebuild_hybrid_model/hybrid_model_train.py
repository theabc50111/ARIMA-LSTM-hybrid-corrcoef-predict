#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
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



# # Prepare data

# In[2]:


# setting of output files
save_raw_corr_data = True
save_train_info_arima_resid_data = True
# data implement setting
data_implement = "sp500_20082017"  # tw50|sp500_20082017|sp500_19972007|tetuan_power
# train set setting
items_setting = "train"  # train|all


# In[3]:


# data loading & implement setting
dataset_path = Path("../dataset/")
if data_implement == "tw50":
    file_name = Path("tw50_hold_20082018_adj_close_pre.csv")
    train_set = ['萬海_adj_close', '豐泰_adj_close', '友達_adj_close', '欣興_adj_close', '台塑化_adj_close', '和泰車_adj_close', '元大金_adj_close', '南電_adj_close', '台塑_adj_close', '統一超_adj_close', '台泥_adj_close', '瑞昱_adj_close', '彰銀_adj_close', '富邦金_adj_close', '研華_adj_close', '中鋼_adj_close', '鴻海_adj_close', '台新金_adj_close', '遠傳_adj_close', '南亞_adj_close', '台達電_adj_close', '台灣大_adj_close', '台化_adj_close', '聯詠_adj_close', '廣達_adj_close', '聯發科_adj_close', '台積電_adj_close', '統一_adj_close', '中信金_adj_close', '長榮_adj_close']
elif data_implement == "sp500_19972007":
    file_name = Path("sp500_hold_19972007_adj_close_pre.csv")
    train_set = ['PXD', 'WAT', 'LH', 'AMGN', 'AOS', 'EFX', 'NEM', 'CTAS', 'MAT', 'VLO', 'APH', 'ADM', 'MLM', 'BK', 'NOV', 'BDX', 'RRC', 'IVZ', 'ED', 'SBUX', 'CI', 'ZION', 'COO', 'FDX', 'GLW', 'GPC', 'HPQ', 'ADI', 'AMG', 'MTB', 'YUM', 'SYK', 'KMX', 'AME', 'BMY', 'KMB', 'JPM', 'AET', 'DLTR', 'MGM', 'FL', 'HD', 'CLX', 'OKE', 'WMB', 'IFF', 'CMS', 'MMC', 'REG', 'ES', 'ITW', 'VRTX', 'QCOM', 'MSI', 'NKTR', 'AMAT', 'BWA', 'ESRX', 'TXT', 'VNO', 'WDC', 'PVH', 'NOC', 'PCAR', 'NSC', 'PHM', 'LUV', 'HUM', 'SPG', 'SJM', 'ABT', 'ALK', 'TAP', 'CAT', 'TMO', 'AES', 'MRK', 'RMD', 'MKC', 'HIG', 'DE', 'ATVI', 'O', 'UNM', 'VMC', 'CMA', 'RHI', 'RE', 'FMC', 'MU', 'CB', 'LNT', 'GE', 'SNA', 'LLY', 'LEN', 'MAA', 'OMC', 'F', 'APA', 'CDNS', 'SLG', 'HP', 'SHW', 'AFL', 'STT', 'PAYX', 'AIG']
elif data_implement in ["sp500_20082017", "paper_eva_1", "paper_eva_2", "paper_eva_3", "paper_eva_4", "paper_eva_5"]:
    file_name = Path("stock08_price.csv")
    train_set = ['CELG', 'PXD', 'WAT', 'LH', 'AMGN', 'AOS', 'EFX', 'CRM', 'NEM', 'JNPR', 'LB', 'CTAS', 'MAT', 'MDLZ', 'VLO', 'APH', 'ADM', 'MLM', 'BK', 'NOV', 'BDX', 'RRC', 'IVZ', 'ED', 'SBUX', 'GRMN', 'CI', 'ZION', 'COO', 'TIF', 'RHT', 'FDX', 'LLL', 'GLW', 'GPN', 'IPGP', 'GPC', 'HPQ', 'ADI', 'AMG', 'MTB', 'YUM', 'SYK', 'KMX', 'AME', 'AAP', 'DAL', 'A', 'MON', 'BRK', 'BMY', 'KMB', 'JPM', 'CCI', 'AET', 'DLTR', 'MGM', 'FL', 'HD', 'CLX', 'OKE', 'UPS', 'WMB', 'IFF', 'CMS', 'ARNC', 'VIAB', 'MMC', 'REG', 'ES', 'ITW', 'NDAQ', 'AIZ', 'VRTX', 'CTL', 'QCOM', 'MSI', 'NKTR', 'AMAT', 'BWA', 'ESRX', 'TXT', 'EXR', 'VNO', 'BBT', 'WDC', 'UAL', 'PVH', 'NOC', 'PCAR', 'NSC', 'UAA', 'FFIV', 'PHM', 'LUV', 'HUM', 'SPG', 'SJM', 'ABT', 'CMG', 'ALK', 'ULTA', 'TMK', 'TAP', 'SCG', 'CAT', 'TMO', 'AES', 'MRK', 'RMD', 'MKC', 'WU', 'ACN', 'HIG', 'TEL', 'DE', 'ATVI', 'O', 'UNM', 'VMC', 'ETFC', 'CMA', 'NRG', 'RHI', 'RE', 'FMC', 'MU', 'CB', 'LNT', 'GE', 'CBS', 'ALGN', 'SNA', 'LLY', 'LEN', 'MAA', 'OMC', 'F', 'APA', 'CDNS', 'SLG', 'HP', 'XLNX', 'SHW', 'AFL', 'STT', 'PAYX', 'AIG', 'FOX', 'MA']
elif data_implement == "tetuan_power":
    file_name = Path("Tetuan City power consumption_pre.csv")
    train_set = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows", "Zone 1 Power Consumption", "Zone 2 Power Consumption", "Zone 3 Power Consumption"]

dataset_df = pd.read_csv(dataset_path/file_name)
dataset_df = dataset_df.set_index('Date')
all_set = list(dataset_df.columns.values[1:])  # all data
test_set = [p for p in all_set if p not in train_set]  # all data - train data
logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

# train set setting
if items_setting == "all":
    items_set = all_set
    output_set_name = "_all"
elif items_setting == "train":
    items_set = train_set
    output_set_name = "_train"
train_info = {"tw50": {"items":items_set, "file_name": "tw50_20082017"},
              "sp500_19972007": {"items":items_set, "file_name": f"sp500_19972007"},
              "sp500_20082017": {"items": items_set, "file_name": f"sp500_20082017"},
              "tetuan_power": {"items": items_set, "file_name":  f"tetuan_power"}}
items_implement = train_info[data_implement]['items']
logging.info(f"===== len(train set): {len(items_implement)} =====")

# setting of name of output files and pictures title
output_file_name = train_info[data_implement]['file_name'] + output_set_name
logging.info(f"===== file_name basis:{output_file_name} =====")

# display(dataset_df)


# In[5]:


def gen_data_corr(items: list, corr_ind: list) -> "pd.DataFrame":
    tmp_corr = dataset_df[items[0]].rolling(window=100).corr(dataset_df[items[1]])
    tmp_corr = tmp_corr.iloc[corr_ind].values
    data_df = pd.DataFrame(tmp_corr.reshape(-1, 24), dtype="float32")
    ind = [f"{items[0]} & {items[1]}_{i}" for i in range(0, 100, 20)]
    data_df.index = ind
    return data_df


def gen_train_data(items: list, corr_ind: list, save_file: bool = False)-> "four pd.DataFrame":
    train_df = pd.DataFrame(dtype="float32")
    dev_df = pd.DataFrame(dtype="float32")
    test1_df = pd.DataFrame(dtype="float32")
    test2_df = pd.DataFrame(dtype="float32")

    for pair in tqdm(combinations(items, 2)):
        data_df = gen_data_corr([pair[0], pair[1]], corr_ind=corr_ind)
        data_split = {'train': [0, 21], 'dev': [1, 22], 'test1': [2, 23], 'test2': [3, 24]}
        train_df = pd.concat([train_df, data_df.iloc[:, 0:21]])
        dev_df = pd.concat([dev_df, data_df.iloc[:, 1:22]])
        test1_df = pd.concat([test1_df, data_df.iloc[:, 2:23]])
        test2_df = pd.concat([test2_df, data_df.iloc[:, 3:24]])

    if save_file:
        before_arima_data_path = dataset_path/f"{output_file_name}_before_arima"
        before_arima_data_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(before_arima_data_path/f"{output_file_name}_train.csv")
        dev_df.to_csv(before_arima_data_path/f"{output_file_name}_dev.csv")
        test1_df.to_csv(before_arima_data_path/f"{output_file_name}_test1.csv")
        test2_df.to_csv(before_arima_data_path/f"{output_file_name}_test2.csv")

    return train_df, dev_df, test1_df, test2_df


before_arima_data_path = dataset_path/f"{output_file_name}_before_arima"
train_df = before_arima_data_path/f"{output_file_name}_train.csv"
dev_df = before_arima_data_path/f"{output_file_name}_dev.csv"
test1_df = before_arima_data_path/f"{output_file_name}_test1.csv"
test2_df = before_arima_data_path/f"{output_file_name}_test2.csv"
if train_df.exists() and dev_df.exists() and test1_df.exists() and test2_df.exists():
    corr_datasets = (pd.read_csv(train_df), pd.read_csv(dev_df), pd.read_csv(test1_df), pd.read_csv(test2_df))
else:
    corr_ind = list(range(99, 2400, 100))  + list(range(99+20, 2500, 100)) + list(range(99+40, 2500, 100)) + list(range(99+60, 2500, 100)) + list(range(99+80, 2500, 100))
    corr_datasets = gen_train_data(items_implement, corr_ind, save_file = save_raw_corr_data)


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
    for corr_pair, corr_series in tqdm(dataset.iterrows()):
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
            arima_infos = [corr_pair, arima_model_name]
            for attr in ["aic", "pvalues", "params", "arparams", "aroots", "maparams", "maroots"]:
                try:
                    val = getattr(arima_model, attr)()
                except AttributeError:
                    arima_infos.append(None)
                else:
                    arima_infos.append(val)
            else:
                arima_model_info_list.append(arima_infos)
        finally:
            find_arima_model = False


    arima_model_info_df = pd.DataFrame(arima_model_info_list, dtype="float32", columns=["items", "model_name", "aic", "pvalues", "params", "arparams", "aroots", "maparams", "maroots"]).set_index("items")
    arima_output_df = pd.DataFrame(arima_output_list, dtype="float32", index=dataset.index)
    arima_resid_df = pd.DataFrame(arima_resid_list, dtype="float32", index=dataset.index)

    if save_file_period:
        after_arima_data_path = dataset_path/f"{output_file_name}_after_arima"
        after_arima_data_path.mkdir(parents=True, exist_ok=True)
        arima_model_info_df.to_csv(after_arima_data_path/f'{output_file_name}_arima_model_info_{save_file_period}.csv')
        arima_output_df.to_csv(after_arima_data_path/f'{output_file_name}_arima_output_{save_file_period}.csv')
        arima_resid_df.to_csv(after_arima_data_path/f'{output_file_name}_arima_resid_{save_file_period}.csv')

    return arima_output_df, arima_resid_df, arima_model_info_df


# In[9]:


after_arima_data_path = dataset_path/f"{output_file_name}_after_arima"
arima_model_info_df = after_arima_data_path/f'{output_file_name}_arima_model_info_test2.csv'
arima_output_df = after_arima_data_path/f'{output_file_name}_arima_output_test2.csv'
arima_resid_df = after_arima_data_path/f'{output_file_name}_arima_resid_test2.csv'
if arima_model_info_df.exists() and arima_output_df.exists() and arima_resid_df.exists():
    pass
else:
    for (file_name, dataset) in tqdm(zip(['train', 'dev', 'test1', 'test2'], corr_datasets)):
        if save_train_info_arima_resid_data:
            arima_model(dataset, save_file_period=file_name)
        else:
            arima_model(dataset)


# # LSTM

# In[ ]:


# Dataset.from_tensor_slices(dict(pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv')))
after_arima_data_path = dataset_path/f"{output_file_name}_after_arima"
lstm_train_X = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_train.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_train_Y = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_train.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_dev_X = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_dev.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_dev_Y = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_dev.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test1_X = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_test1.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test1_Y = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_test1.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test2_X = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_test2.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test2_Y = pd.read_csv(after_arima_data_path/f'{output_file_name}_arima_resid_test2.csv').set_index('Unnamed: 0').iloc[::, -1]

lstm_train_X = lstm_train_X.values.reshape(-1, 20, 1)
lstm_train_Y = lstm_train_Y.values.reshape(-1, 1)
lstm_dev_X = lstm_dev_X.values.reshape(-1, 20, 1)
lstm_dev_Y = lstm_dev_Y.values.reshape(-1, 1)
lstm_test1_X = lstm_test1_X.values.reshape(-1, 20, 1)
lstm_test1_Y = lstm_test1_Y.values.reshape(-1, 1)
lstm_test2_X = lstm_test2_X.values.reshape(-1, 20, 1)
lstm_test2_Y = lstm_test2_Y.values.reshape(-1, 1)


# In[ ]:


def double_tanh(x):
    return (tf.math.tanh(x) *2)


def build_many_one_lstm():
    inputs = Input(shape=(20, 1))
    lstm_1 = LSTM(units=10, kernel_regularizer=l1_l2(0.2, 0.0), bias_regularizer=l1_l2(0.2, 0.0), activation="tanh", dropout=0.1)(inputs)
    outputs = Dense(units=1, activation=double_tanh)(lstm_1)
    return keras.Model(inputs, outputs, name="many_one_lstm")


opt = keras.optimizers.Adam(learning_rate=0.001)
lstm_model = build_many_one_lstm()
lstm_model.summary()
lstm_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])


# In[ ]:


model_dir = Path('./models/')
log_dir = Path('./models/lstm_train_logs/')
res_dir = Path('./results/')
model_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)
res_csv_path = res_dir/f'{output_file_name}_LSTM_evaluation.csv'
res_csv_path.touch(exist_ok=True)
with open(res_csv_path, 'r+') as f:
    if not f.read():
        f.write("epoch,TRAIN_MSE,DEV_MSE,TEST1_MSE,TEST2_MSE,TRAIN_MAE,DEV_MAE,TEST1_MAE,TEST2_MAE")

res_df = pd.read_csv(res_csv_path)
saved_model_list = [int(p.stem.split('_')[1]) for p in model_dir.glob('*.h5')]
model_cbk = TensorBoard(log_dir=log_dir)
epoch_start = max(saved_model_list) if saved_model_list else 1
max_epoch = 300
batch_size = 64

for epoch_num in tqdm(range(epoch_start, max_epoch)):
    if epoch_num > 1:
        lstm_model = load_model(model_dir/f"{output_file_name}_epoch_{epoch_num - 1}.h5", custom_objects={'double_tanh':double_tanh})

    save_model = ModelCheckpoint(model_dir/f"{output_file_name}_epoch_{epoch_num}.h5",
                                                 monitor='loss', verbose=1, mode='min', save_best_only=False)
    lstm_model.fit(lstm_train_X, lstm_train_Y, epochs=1, batch_size=batch_size, shuffle=True, callbacks=[model_cbk, save_model])

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


# In[ ]:




