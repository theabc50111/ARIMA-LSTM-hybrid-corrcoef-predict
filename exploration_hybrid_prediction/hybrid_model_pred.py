#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import math
import os
from pathlib import Path
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression 
from pmdarima.arima import ARIMA, auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense
import warnings
import logging

warnings.simplefilter("ignore")
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO)
err_log_handler = logging.FileHandler(filename="./results/arima_train_err_log.txt", mode='a')
err_logger = logging.getLogger("arima_train_err")
err_logger.addHandler(err_log_handler)



# # Prepare data

# In[ ]:


# setting of output files
save_raw_corr_data = True
save_arima_resid_data = True
# data implement setting
data_implement = "sp500_20082017"  # tw50|sp500_20082017|sp500_19972007|tetuan_power
                                # |paper_eva_1|paper_eva_2|paper_eva_3|paper_eva_4|paper_eva_5
# lstm weight setting
lstm_weight_setting = "sp500_20082017"  # tw50|sp500_20082017|sp500_19972007|tetuan_power
# evaluation set setting
items_setting = "test"  # test|all
# data time period setting
time_period = "_test2"


# In[ ]:


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

# evaluation set setting
if items_setting == "all":
    items_set = all_set
    output_set_name = "_all"
elif items_setting == "test":
    items_set = test_set
    output_set_name = "_test"

evaluation_info = {"paper_eva_1": {"items": ['PRGO', 'MRO', 'ADP', 'HCP', 'FITB', 'PEG', 'SYMC', 'EOG', 'MDT', 'NI'], "file_name": "paper_eva_1_res"},
                   "paper_eva_2": {"items": ['STI', 'COP', 'MCD', 'AON', 'JBHT', 'DISH', 'GS', 'LRCX', 'CTXS', 'LEG'], "file_name": "paper_eva_2_res"},
                   "paper_eva_3": {"items": ['TJX', 'EMN', 'JCI', 'C', 'BIIB', 'HOG', 'PX', 'PH', 'XEC', 'JEC'], "file_name": "paper_eva_3_res"},
                   "paper_eva_4": {"items": ['ROP', 'AZO', 'URI', 'TROW', 'CMCSA', 'SLB', 'VZ', 'MAC', 'ADS', 'MCK'], "file_name": "paper_eva_4_res"},
                   "paper_eva_5": {"items": ['RL', 'CVX', 'SRE', 'PFE', 'PCG', 'UTX', 'NTRS', 'INCY', 'COP', 'HRL'], "file_name": "paper_eva_5_res"},
                   "tw50": {"items": items_set, "file_name": f"tw50_20082017_res"},
                   "sp500_19972007": {"items": items_set, "file_name": f"sp500_19972007_res"},
                   "sp500_20082017": {"items": items_set, "file_name": f"sp500_20082017_res"},
                   "tetuan_power": {"items": items_set, "file_name":  f"tetuan_power_res"}}
items_implement = evaluation_info[data_implement]['items']
logging.info(f"===== len(evaluation set): {len(items_implement)} =====")

# lstm weight setting
lstm_weight_path = Path("../rebuild_hybrid_model/models/")
if lstm_weight_setting == "sp500_20082017":
    # lstm_weight_filepath = lstm_weight_path/"20220909/sp500_20082017_train_res_epoch_43.h5"
    lstm_weight_filepath = lstm_weight_path/"20220924/Kristian_Senneset_LSTM_hyper_params/sp500_20082017_train_epoch_146.h5"
    lstm_weight_name = "_sp500_20082017LSTM"
elif lstm_weight_setting == "tw50":
    lstm_weight_filepath = lstm_weight_path/"20220816/tw50_20082017_epoch_246.h5"
    lstm_weight_name = "_tw50LSTM"
elif lstm_weight_setting == "tetuan_power":
    lstm_weight_filepath = lstm_weight_path/"20220831/tetuan_power_res_epoch_597.h5"
    lstm_weight_name = "_tetuan_powerLSTM"
logging.info(f"===== LSTM weight:{lstm_weight_name} =====")

# setting of name of output files and pictures title
output_file_name = evaluation_info[data_implement]['file_name'] + output_set_name + time_period + lstm_weight_name
fig_title = data_implement + output_set_name + time_period + lstm_weight_name
logging.info(f"===== file_name basis:{output_file_name}, fig_title basis:{fig_title} =====")



def gen_unseen_data_corr(items: list, time_period:str = "_test2", ret_date: bool = False) -> "pd.DataFrame, pd.Series | pd.DataFrame":
    tmp_corr = dataset_df[items[0]].rolling(window=100).corr(dataset_df[items[1]])
    tmp_corr = tmp_corr.iloc[99::100]
    if time_period == "_test2":
        corr_series = tmp_corr[3:24] # correspond to test2_dataset of original paper
    elif time_period == "_test1" :
        corr_series = tmp_corr[2:23] # correspond to test1_dataset of original paper
    elif time_period == "_dev":
        corr_series = tmp_corr[1:22] # correspond to dev_dataset of original paper
    elif time_period == "_train":
        corr_series = tmp_corr[:21] # correspond to train_dataset of original papaer 
    unseen_data_df = pd.DataFrame(corr_series).reset_index().drop(['Date'], axis=1).T
    if ret_date:
        return unseen_data_df, corr_series
    else:
        return unseen_data_df



# # ARIMA model


def arima_model(dataset: "pd.DataFrame", portfolio: list, overview: bool = False) -> ("np.array", "pd.DataFrame", str):
    model_110 = ARIMA(order=(1, 1, 0), out_of_sample_size=10, mle_regression=True, suppress_warnings=True)
    model_011 = ARIMA(order=(0, 1, 1), out_of_sample_size=10, mle_regression=True, suppress_warnings=True)
    model_111 = ARIMA(order=(1, 1, 1), out_of_sample_size=10, mle_regression=True, suppress_warnings=True)
    model_211 = ARIMA(order=(2, 1, 1), out_of_sample_size=10, mle_regression=True, suppress_warnings=True)
    model_210 = ARIMA(order=(2, 1, 0), out_of_sample_size=10, mle_regression=True, suppress_warnings=True)
    # model_330 = ARIMA(order=(3, 3, 0), out_of_sample_size=0, mle_regression=True, suppress_warnings=True)

    model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210}
    # model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111}

    tested_models = []
    arima_model = None
    arima_attr_list = ["aic", "arparams", "aroots", "maparams", "maroots", "params", "pvalues"]
    arima_infos = dict(zip(arima_attr_list, [None]*len(arima_attr_list)))
    find_arima_model = False
    for _, corr_series in dataset.iterrows():
        while not find_arima_model:
            try:
                for model_key in model_dict:
                    if model_key not in tested_models:
                        test_model = model_dict[model_key].fit(corr_series[:-1])  # only use first 20 corrletaion coefficient to fit ARIMA model
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
                    err_logger.error(f"fatal error, {portfolio} doesn't have appropriate arima model\n", exc_info=True)
                    raise NotImplementedError(f"fatal error, {portfolio} doesn't have appropriate arima model\n")
            else:
                # model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210, "model_330": model_330}
                model_dict = {"model_110": model_110, "model_011": model_011, "model_111": model_111, "model_211": model_211, "model_210": model_210}
                tested_models.clear()
                find_arima_model = True
        try:
            arima_pred = list(arima_model.predict(n_periods=1))
        except Exception:
            err_logger.error(f"{portfolio} in {time_period} be predicted by {arima_model_name}(its aic:{arima_model.aic()}) getting error:\n", exc_info=True)
            raise NotImplementedError(f"{portfolio} in {time_period} be predicted by {arima_model_name}(its aic:{arima_model.aic()}) getting error\n")
        else:
            arima_pred_in_sample = list(arima_model.predict_in_sample())
            arima_pred_in_sample = [np.mean(arima_pred_in_sample[1:])] + arima_pred_in_sample[1:]
            arima_output = arima_pred_in_sample + arima_pred
            arima_output = np.clip(np.array(arima_output), -1, 1)

            arima_resid = pd.Series(np.array(corr_series) - arima_output).iloc[:-1]

            for attr in arima_infos.keys():
                try:
                    arima_infos[attr] = getattr(arima_model, attr)()
                except AttributeError:
                    pass
        finally:
            find_arima_model = False
    if overview:
        plt.plot(arima_output, label="arima_pred")
        plt.plot(dataset.T, label="data")
        plt.plot(arima_resid, label="res")
        plt.legend()
        plt.show()
        plt.close()

    return arima_output, arima_resid, arima_model_name, *[v for k, v in sorted(arima_infos.items(), key=lambda x:x[0])]


# # LSTM


def double_tanh(x):
    return (tf.math.tanh(x) * 2)


lstm_model = load_model(lstm_weight_filepath, custom_objects={'double_tanh':double_tanh})


# # Hybrid model

# In[ ]:


def stl_decompn(corr_series: "pd.Series", overview: bool = False) -> (float, float, float):
    output_resid = 100000
    output_trend = None
    output_period = None
    for p in range(2, 11):
        decompose_result_mult = seasonal_decompose(corr_series, period=p)
        resid_sum = np.abs(decompose_result_mult.resid).sum()
        if output_resid > resid_sum:
            output_resid = resid_sum
            output_trend = decompose_result_mult.trend.dropna()
            output_period = p

    reg = LinearRegression().fit(np.arange(len(output_trend)).reshape(-1, 1), output_trend)

    if overview:
        decompose_result_mult = seasonal_decompose(corr_series, period=output_period)
        trend = decompose_result_mult.trend.dropna().reset_index(drop=True)
        plt.figure(figsize=(7, 1))
        plt.plot(trend)
        plt.plot([0, len(trend)], [reg.intercept_, reg.intercept_+len(trend)*reg.coef_])
        plt.title("trend & regression line")
        plt.show()
        plt.close()
        decompose_result_mult.plot()
        plt.show()
        plt.close()

    return output_period, output_resid, output_trend.std(), reg.coef_[0]


# In[ ]:


unseen_data_corr_df, unseen_data_corr_series = gen_unseen_data_corr([items_implement[0], items_implement[1]], ret_date=True)
stl_decompn(unseen_data_corr_series, overview=True)


# In[ ]:


res_list = []
unseen_data_corr_df_concat = pd.DataFrame(columns=list(range(21))+['items'])
unseen_data_arima_resid_concat = pd.DataFrame(columns=list(range(20))+['items'])
count = 0
for items in tqdm(combinations(items_implement, 2)):
    unseen_data_corr_df, unseen_data_corr_series = gen_unseen_data_corr(items, time_period=time_period, ret_date=True)
    try:
        arima_pred, residual, arima_model_name, arima_aic, arima_arparams, arima_aroots, arima_maparams, arima_maroots, arima_params, arima_pvalues = arima_model(unseen_data_corr_df, items)
    except NotImplementedError:
        continue
    else:
        unseen_res = residual.values.reshape((-1, 20, 1))
        lstm_pred = lstm_model.predict(unseen_res)
        season_period, stl_resid, stl_trend_std, coef_reg_trend = stl_decompn(unseen_data_corr_series)
        items_res_dic = {"items": f"{items[0]} & {items[1]}",
                         "corr_ser_mean": unseen_data_corr_series.mean(),
                         "corr_ser_std": unseen_data_corr_series.std(),
                         "corr_season_period": season_period,
                         "corr_stl_resid": stl_resid,
                         "corr_stl_trend_std": stl_trend_std,
                         "corr_trend_coef": coef_reg_trend,
                         "arima_model": arima_model_name,
                         "arima_aic": arima_aic,
                         "arima_arparams": arima_arparams,
                         "arima_aroots": arima_aroots,
                         "arima_maparams": arima_maparams,
                         "arima_maroots": arima_maroots,
                         "arima_params": arima_params,
                         "arima_pvalues": arima_pvalues,
                         "lstm_pred": lstm_pred[0][0],
                         "arima_pred": arima_pred[-1],
                         "ground_truth": unseen_data_corr_df.iloc[0, -1]}

        res_list.append(items_res_dic)
        unseen_data_corr_df['items'] = f"{items[0]} & {items[1]}"
        unseen_data_corr_df_concat = pd.concat([unseen_data_corr_df_concat, unseen_data_corr_df])
        residual['items'] = f"{items[0]} & {items[1]}"
        unseen_data_arima_resid_concat = pd.concat([unseen_data_arima_resid_concat, residual])

if save_raw_corr_data:
    unseen_data_corr_df_concat = unseen_data_corr_df_concat.set_index('items')
    unseen_data_corr_df_concat.to_csv(f"./results/{output_file_name}_raw_corr.csv", index=True)

if save_arima_resid_data:
    unseen_data_arima_resid_concat = unseen_data_arima_resid_concat.set_index('items')
    unseen_data_arima_resid_concat.to_csv(f"./results/{output_file_name}_arima_resid.csv", index=True)

res_df = pd.DataFrame(res_list)
res_df.to_csv(f"./results/{output_file_name}.csv", index=False)


# # Display results


res_df = pd.read_csv(f"./results/{output_file_name}.csv")


def res_df_postprocess(target_df: pd.core.frame.DataFrame) -> None:
    target_df["hybrid_model_pred"] =  target_df["arima_pred"] + target_df["lstm_pred"]
    target_df["error"] = target_df["ground_truth"] - target_df["hybrid_model_pred"]
    target_df["absolute_err"] = target_df["error"].abs()
    target_df['arima_pred_dir'] = np.sign(target_df['ground_truth'] * target_df['arima_pred'])
    target_df['arima_err'] = target_df['ground_truth'] - target_df['arima_pred']
    target_df["lstm_compensation_dir"] = np.sign(target_df['arima_err']) * np.sign(target_df['lstm_pred'])
    quantile_mask = np.logical_and(res_df['error'] < np.quantile(res_df['error'], 0.75), res_df['error'] > np.quantile(res_df['error'], 0.25)).tolist()
    target_df['high_pred_performance'] = quantile_mask
    target_df['items[0]'] = target_df.apply(lambda row:row['items'].split(" & ")[0], axis=1)
    target_df['items[1]'] = target_df.apply(lambda row:row['items'].split(" & ")[1], axis=1)


res_df_postprocess(res_df)


def plot_exploration(target_df: pd.core.frame.DataFrame, title: str) -> None:
    fig, axes = plt.subplots(figsize=(20, 20), nrows=7, ncols=2, sharex=False, sharey=False, dpi=100)
    s0 = axes[0, 0]
    s0.set_title("ABS_err violin")
    sns.violinplot(y=target_df["absolute_err"], ax=s0)
    s1 = axes[0, 1]
    s1.set_title("Err violin")
    sns.violinplot(y=target_df["error"], ax=s1)
    s2 = axes[1, 0]
    s2.set_title("ABS_err hist")
    target_df['absolute_err'].hist(bins=[b/10 for b in range(11)], ax=s2)
    s3 = axes[1, 1]
    s3.set_title("Err hist")
    target_df['error'].hist(bins=[b/10 for b in range(-10, 11)], ax=s3)
    s4 = axes[2, 0]
    s4.set_title("LSTM_compensation_dir count")
    sns.countplot(x="lstm_compensation_dir", data=target_df, ax=s4)
    s5 = axes[2, 1]
    s5.set_title("LSTM_compensation_dir count groupby ARIMA_pred_dir")
    df_gb = target_df.groupby(['arima_pred_dir', 'lstm_compensation_dir']).size().unstack(level=1)
    df_gb.plot(kind='bar', ax=s5)
    s6 = axes[3, 0]
    s6.set_title("ARIMA_model prediction Err violin group by LSTM_compensation_dir")
    sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["arima_err"], ax=s6)
    s8 = axes[4, 0]
    s8.set_title("ARIMA_model prediction magnitude group by LSTM_compensation_dir")
    sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["arima_pred"], ax=s8)
    s9 = axes[4, 1]
    s9.set_title("LSTM compensation magnitude group by LSTM_compensation_dir")
    sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["lstm_pred"], ax=s9)
    s10 = axes[5, 0]
    s10.set_title("Correlation magnitude in last period group by LSTM_compensation_dir")
    sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["ground_truth"], ax=s10)
    s11 = axes[5, 1]
    s11.set_title("Hybrid Err violin group by LSTM_compensation_dir")
    sns.violinplot(x=target_df["lstm_compensation_dir"], y=target_df["error"], ax=s11)
    s12 = axes[6,0]
    s12.set_title("LSTM_compensation_dir pie with wrong ARIMA_pred_dir")
    df_gb.loc[df_gb.index==-1, :].squeeze().plot(kind="pie", autopct='%1.1f%%', ax=s12)
    s13 = axes[6,1]
    s13.set_title("LSTM_compensation_dir pie with correct ARIMA_pred_dir")
    df_gb.loc[df_gb.index==1, :].squeeze().plot(kind="pie", autopct='%1.1f%%', ax=s13)

    fig.suptitle(f"{title}_basic_exploration")
    plt.tight_layout()
    plt.savefig(f"./results/hybrid_prediction_analysis_{title}.png")
    plt.show()
    plt.close()


# In[ ]:


plot_exploration(res_df, fig_title)


# ---

# In[ ]:


def plot_exploration_pred_perform(target_df: pd.core.frame.DataFrame, title: str) -> None:
    fig, axes = plt.subplots(figsize=(20, 20), nrows=6, ncols=2, sharex=False, sharey=False, dpi=100)
    s1 = axes[0, 0]
    s1.set_title("LSTM_compensation_dir count groupby prediction performance")
    df_gb = target_df.groupby(['high_pred_performance', 'lstm_compensation_dir']).size().unstack(level=1)
    df_gb.plot(kind='bar', ax=s1)
    s2 = axes[0, 1]
    s2.set_title("ARIMA_model prediction magnitude group by prediction performance")
    sns.violinplot(x=target_df["high_pred_performance"], y=target_df["arima_pred"], ax=s2)
    s3 = axes[1, 0]
    s3.set_title("LSTM compensation magnitude group by prediction performance")
    sns.violinplot(x=target_df["high_pred_performance"], y=target_df["lstm_pred"], ax=s3)
    s4 = axes[1, 1]
    s4.set_title("Correlation magnitude in last period group by prediction performance")
    sns.violinplot(x=target_df["high_pred_performance"], y=target_df["ground_truth"], ax=s4)
    s5 = axes[2, 0]
    s5.set_title("Correlation series mean groupby prediction performance")
    sns.violinplot(x=target_df['high_pred_performance'], y=target_df["corr_ser_mean"], ax=s5)
    s6 = axes[2, 1]
    s6.set_title("Correlation series std groupby prediction performance")
    sns.violinplot(x=target_df['high_pred_performance'], y=target_df["corr_ser_std"], ax=s6)
    s7 = axes[3, 0]
    s7.set_title("Correlation series stl_period group by prediction performance")
    sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_season_period"], ax=s7)
    s8 = axes[3, 1]
    s8.set_title("Correlation series stl_residual group by prediction performance")
    sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_stl_resid"], ax=s8)
    s9 = axes[4, 0]
    s9.set_title("Correlation series stl_trend_std group by prediction performance")
    sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_stl_trend_std"], ax=s9)
    s10 = axes[4, 1]
    s10.set_title("Correlation series stl_trend_coef group by prediction performance")
    sns.violinplot(x=target_df["high_pred_performance"], y=target_df["corr_trend_coef"], ax=s10)
    s11 = axes[5, 0]
    s11.set_title("ARIMA_pred_dir count groupby prediction performance")
    df_gb = target_df.groupby(['high_pred_performance', 'arima_pred_dir']).size().unstack(level=1)
    df_gb.plot(kind='bar', ax=s11)

    fig.suptitle(F"{title}_groupby prediction")
    plt.tight_layout()
    plt.savefig(f"./results/hybrid_prediction_analysis_groupby_pred_perform_{title}.png")
    plt.show()
    plt.close()


# In[ ]:


def plot_stock_freq(target_df: pd.core.frame.DataFrame, title: str) -> None:
    stocks_show_freq = target_df.loc[target_df['high_pred_performance'] == True, ['items[0]','items[1]']].stack().value_counts().to_dict()
    plt.figure(figsize=(80, 10), dpi=100)
    plt.bar(range(len(stocks_show_freq)), list(stocks_show_freq.values()))
    plt.xticks(range(len(stocks_show_freq)), list(stocks_show_freq.keys()), rotation=60)
    plt.title(F"{title}_items appearence frequence")
    plt.savefig(f"./results/items_appearence_frequence_{title}.png")
    plt.show()
    plt.close()


# In[ ]:


print(f"mse :{(res_df['error']**2).mean()}",
        f"std of square_err :{(res_df['error']**2).std()}",
        f"rmse :{np.sqrt((res_df['error']**2).mean())}",
        f"mae : {res_df['absolute_err'].mean()}",
        f"std of abs_err: {res_df['absolute_err'].std()}",
        f"sklearn mse: {mean_squared_error(res_df['ground_truth'], res_df['hybrid_model_pred'])}")
print("-"*50)
print(f"mse of ARIMA :{(res_df['arima_err']**2).mean()}",
        f"std of square_err ARIMA :{(res_df['arima_err']**2).std()}",
        f"rmse of ARIMA :{np.sqrt((res_df['arima_err']**2).mean())}",
        f"sklearn mse of ARIMA: {mean_squared_error(res_df['ground_truth'], res_df['arima_err'])}")


# In[ ]:


plot_exploration_pred_perform(res_df, fig_title)


# In[ ]:


plot_stock_freq(res_df, fig_title)
