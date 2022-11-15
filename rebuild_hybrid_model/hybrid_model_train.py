#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from itertools import combinations
from pathlib import Path
import sys
import warnings
import logging
from pprint import pformat
import traceback

import numpy as np
import pandas as pd
from pmdarima.arima import ARIMA, auto_arima
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l1_l2
import dynamic_yaml
import yaml

sys.path.append("/tf/correlation-coef-predict/ywt_library")
import data_generation
from data_generation import data_gen_cfg

with open('../config/data_config.yaml') as f:
    data = dynamic_yaml.load(f)
    data_cfg = yaml.full_load(dynamic_yaml.dump(data))

warnings.simplefilter("ignore")
logging.basicConfig(level=logging.INFO)
err_log_handler = logging.FileHandler(filename="./save_models/arima_train_err_log.txt", mode='a')
err_logger = logging.getLogger("arima_train_err")
err_logger.addHandler(err_log_handler)

# %load_ext pycodestyle_magic
# %pycodestyle_on --ignore E501
logging.info(pformat(data_cfg, indent=1, width=100, compact=True))


# # Prepare data

# ## Data implement & output setting & trainset setting

# In[ ]:


model_dir = Path('./save_models/')
lstm_log_dir = Path('./save_models/lstm_train_logs/')
res_dir = Path('./results/')
model_dir.mkdir(parents=True, exist_ok=True)
lstm_log_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

# setting of output files
save_corr_data = True
save_arima_resid_data = True
# data implement setting
data_implement = "SP500_20082017"  # ['BITCOIN_NVDA', 'PAPER_EVA_1', 'PAPER_EVA_2', 'PAPER_EVA_3',
                                                          # 'PAPER_EVA_4', 'PAPER_EVA_5', 'SP500_19972007', 'SP500_20082017',
                                                          # 'SP500_20082017_CONSUMER_DISCRETIONARY', 'TEST_CASE', TETUAN_POWER', 'TW50_20082018']
# train set setting
train_items_setting = "-train_train"  # -train_train|-train_all
# data split  period setting, only suit for only settings of Korean paper
data_split_settings = ["-data_sp_train", "-data_sp_dev", "-data_sp_test1", "-data_sp_test2", ]
# lstm_hyper_params
lstm_hyper_param = "-kS_hyper"


# In[ ]:


# data loading & implement setting
dataset_df = pd.read_csv(data_cfg["DATASETS"][data_implement]['FILE_PATH'])
dataset_df = dataset_df.set_index('Date')
all_set = list(dataset_df.columns)  # all data
train_set = data_cfg["DATASETS"][data_implement]['TRAIN_SET']
test_set = data_cfg['DATASETS'][data_implement]['TEST_SET'] if data_cfg['DATASETS'][data_implement].get('TEST_SET') else [p for p in all_set if p not in train_set]  # all data - train data
logging.info(f"===== len(train_set): {len(train_set)}, len(all_set): {len(all_set)}, len(test_set): {len(test_set)} =====")

# train items implement settings
items_implement = train_set if train_items_setting == "-train_train" else all_set
logging.info(f"===== len(train set): {len(items_implement)} =====")

# setting of name of output files and pictures title
output_file_name = data_cfg["DATASETS"][data_implement]['OUTPUT_FILE_NAME_BASIS'] + train_items_setting
logging.info(f"===== file_name basis:{output_file_name} =====")

# display(dataset_df)


# ## Randon pick items for trainset # Not always necessary to operate

# In[ ]:


# import pprint
# import random

# random.seed(10)


# all_items = pd.read_csv(dataset_path/file_name).set_index("Date").columns.to_list()
# train_set = random.sample(all_items, len(all_items)-10)
# print(len(train_set))
# pp = pprint.PrettyPrinter(width=500, compact=True)
# pp.pprint(train_set)


# ## Load or Create Correlation Data

# In[ ]:


corr_data_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-corr_data"
corr_data_dir.mkdir(parents=True, exist_ok=True)
data_length = int(len(dataset_df)/data_gen_cfg["CORR_WINDOW"])*data_gen_cfg["CORR_WINDOW"]
corr_ser_len_max = int((data_length-data_gen_cfg["CORR_WINDOW"])/data_gen_cfg["CORR_STRIDE"])

train_df_path = corr_data_dir/f"{output_file_name}-corr_train.csv"
dev_df_path = corr_data_dir/f"{output_file_name}-corr_dev.csv"
test1_df_path = corr_data_dir/f"{output_file_name}-corr_test1.csv"
test2_df_path = corr_data_dir/f"{output_file_name}-corr_test2.csv"
all_corr_df_paths = dict(zip(["train_df", "dev_df", "test1_df", "test2_df"],
                             [train_df_path, dev_df_path, test1_df_path, test2_df_path]))
if all([df_path.exists() for df_path in all_corr_df_paths.values()]):
    corr_datasets = [pd.read_csv(df_path).set_index("Unnamed: 0") for df_path in all_corr_df_paths.values()]
else:
    corr_datasets = data_generation.gen_train_data(items_implement, raw_data_df=dataset_df, corr_df_paths=all_corr_df_paths, corr_ser_len_max=corr_ser_len_max, save_file=save_corr_data)


# # ARIMA model

# In[ ]:


def arima_model(dataset: "pd.DataFrame", arima_result_dir: "pathlib.PosixPath", data_split_setting: str = "", save_file: bool = False) -> ("pd.DataFrame", "pd.DataFrame", "pd.DataFrame"):
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
            err_logger.error(f"{corr_pair} in {data_split_setting} be predicted by {arima_model_name}(its aic:{arima_model.aic()}) getting error:\n", exc_info=True)
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

    if save_file:
        arima_model_info_df.to_csv(arima_result_dir/f'{output_file_name}-arima_model_info{data_split_setting}.csv')
        arima_output_df.to_csv(arima_result_dir/f'{output_file_name}-arima_output{data_split_setting}.csv')
        arima_resid_df.to_csv(arima_result_dir/f'{output_file_name}-arima_resid{data_split_setting}.csv')

    return arima_output_df, arima_resid_df, arima_model_info_df


# In[ ]:


arima_result_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-arima_res"
arima_result_dir.mkdir(parents=True, exist_ok=True)
arima_result_paths = []
arima_result_types = ["-arima_model_info", "-arima_output", "-arima_resid"]

for data_sp_setting in data_split_settings:
    for arima_result_type in arima_result_types:
        arima_result_paths.append(arima_result_dir/f'{output_file_name}{arima_result_type}{data_sp_setting}.csv')

if all([df_path.exists() for df_path in arima_result_paths]):
    pass
else:
    for (data_sp_setting, dataset) in tqdm(zip(data_split_settings, corr_datasets)):
        arima_model(dataset, arima_result_dir=arima_result_dir, data_split_setting=data_sp_setting, save_file=save_arima_resid_data)


# In[ ]:


# arima_result_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-arima_res"
# arima_result_dir.mkdir(parents=True, exist_ok=True)
# # after_arima_data_path = dataset_path/f"{output_file_name}_after_arima"
# arima_result_dfs = []
# for data_sp_setting in data_split_settings:
#     for arima_result_type in ["-arima_model_info", "-arima_output", "-arima_resid"]:
#         arima_result_dfs.append(arima_result_dir/f'{output_file_name}{arima_result_type}{data_sp_setting}.csv')
#     # arima_model_info_df_path = arima_result_dir/f'{output_file_name}-arima_model_info{data_sp_setting}.csv'
#     # arima_output_df_path = arima_result_dir/f'{output_file_name}-arima_output{data_sp_setting}.csv'
#     # arima_resid_df_path = arima_result_dir/f'{output_file_name}-arima_resid{data_sp_setting}.csv'
# if arima_model_info_df.exists() and arima_output_df.exists() and arima_resid_df.exists():
#     pass
# else:
#     for (file_name, dataset) in tqdm(zip(data_split_settings, corr_datasets)):
#         if save_arima_resid_data:
#             arima_model(dataset, save_file_period=file_name)
#         else:
#             arima_model(dataset)


# # LSTM

# ## settings of input data of LSTM

# In[ ]:


# Dataset.from_tensor_slices(dict(pd.read_csv(f'./dataset/after_arima/arima_resid_train.csv')))
arima_result_dir = Path(data_cfg["DIRS"]["PIPELINE_DATA_DIR"])/f"{output_file_name}-arima_res"
lstm_train_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_train.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_train_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_train.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_dev_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_dev.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_dev_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_dev.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test1_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test1.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test1_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test1.csv').set_index('Unnamed: 0').iloc[::, -1]
lstm_test2_X = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test2.csv').set_index('Unnamed: 0').iloc[::, :-1]
lstm_test2_Y = pd.read_csv(arima_result_dir/f'{output_file_name}-arima_resid-data_sp_test2.csv').set_index('Unnamed: 0').iloc[::, -1]

lstm_train_X = lstm_train_X.values.reshape(-1, 20, 1)
lstm_train_Y = lstm_train_Y.values.reshape(-1, 1)
lstm_dev_X = lstm_dev_X.values.reshape(-1, 20, 1)
lstm_dev_Y = lstm_dev_Y.values.reshape(-1, 1)
lstm_test1_X = lstm_test1_X.values.reshape(-1, 20, 1)
lstm_test1_Y = lstm_test1_Y.values.reshape(-1, 1)
lstm_test2_X = lstm_test2_X.values.reshape(-1, 20, 1)
lstm_test2_Y = lstm_test2_Y.values.reshape(-1, 1)


# ## settings of LSTM

# In[ ]:


if lstm_hyper_param == "-kS_hyper":
    lstm_layer = LSTM(units=10, kernel_regularizer=l1_l2(0.2, 0.0), bias_regularizer=l1_l2(0.2, 0.0), activation="tanh", dropout=0.1)  # LSTM hyper params from 【Something Old, Something New — A Hybrid Approach with ARIMA and LSTM to Increase Portfolio Stability】


# In[ ]:


def double_tanh(x):
    return (tf.math.tanh(x) *2)


def build_many_one_lstm():
    inputs = Input(shape=(20, 1))
    lstm_1 = lstm_layer(inputs)
    outputs = Dense(units=1, activation=double_tanh)(lstm_1)
    return keras.Model(inputs, outputs, name="many_one_lstm")


opt = keras.optimizers.Adam(learning_rate=0.0001)
lstm_model = build_many_one_lstm()
lstm_model.summary()
lstm_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse', 'mae'])


# In[ ]:


res_csv_path = res_dir/f'{output_file_name}{lstm_hyper_param}_lstm_evaluation.csv'
res_csv_path.touch(exist_ok=True)
with open(res_csv_path, 'r+') as f:
    if not f.read():
        f.write("epoch,TRAIN_MSE,DEV_MSE,TEST1_MSE,TEST2_MSE,TRAIN_MAE,DEV_MAE,TEST1_MAE,TEST2_MAE")

res_df = pd.read_csv(res_csv_path)
saved_model_list = [int(p.stem[p.stem.find("epoch")+len("epoch"):]) for p in model_dir.glob('*.h5')]
model_cbk = TensorBoard(log_dir=lstm_log_dir)
epoch_start = max(saved_model_list) if saved_model_list else 1
max_epoch = 5000
batch_size = 64

try:
    for epoch_num in tqdm(range(epoch_start, max_epoch)):
        if epoch_num > 1:
            lstm_model = load_model(model_dir/f"{output_file_name}{lstm_hyper_param}-epoch{epoch_num - 1}.h5", custom_objects={'double_tanh':double_tanh})

        save_model = ModelCheckpoint(model_dir/f"{output_file_name}{lstm_hyper_param}-epoch{epoch_num}.h5",
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
                                                 "TEST1_MAE", "TEST2_MAE"])
        res_df = pd.concat([res_df, res_each_epoch_df])
        if (res_df.shape[0] % 100) == 0:
            res_df.to_csv(res_csv_path, index=False)  # insurance for 『finally』 part doesent'work
except Exception as e:
    error_class = e.__class__.__name__  # 取得錯誤類型
    detail = e.args[0]  # 取得詳細內容
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    last_call_stack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
    file_name = last_call_stack[0]  # 取得發生的檔案名稱
    line_num = last_call_stack[1]  # 取得發生的行號
    func_name = last_call_stack[2]  # 取得發生的函數名稱
    err_msg = "File \"{}\", line {}, in {}: [{}] {}".format(file_name, line_num, func_name, error_class, detail)
    print(err_msg)
else:
    pass
finally:
    res_df.to_csv(res_csv_path, index=False)


# In[ ]:





# In[ ]:




