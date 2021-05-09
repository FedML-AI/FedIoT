# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:06:52 2021

@author: Mth
"""
import numpy as np
import pandas as pd
import os

cwd = os.getcwd()

# loading path is where the raw data is stored
load_path = os.path.join(cwd, 'data_separated/Samsung_SNH_Webcam')

# saved path is to which the integrated data are stored
saved_path = os.path.join(cwd, 'data_debug/1_pc/test')

gafgyt_path = os.path.join(load_path, 'gafgyt_attacks')
mirai_path = os.path.join(load_path, 'mirai_attacks')

# read benign data
# df_benign = pd.read_csv(os.path.join(load_path, 'benign_traffic.csv'))

# find whether the missing values exist
# if df_benign.isnull().any().any():
#     df_benign.fillna(0)

# read malicious samples from gafgyt and attacks
# except tcp and udp from gafgyt
# from gafgyt attacks
os.chdir(gafgyt_path)
gafgyt_list = os.listdir()
gafgyt = []
for i in range(len(gafgyt_list)):
    gafgyt.append(pd.read_csv(gafgyt_list[i]))\

# random sample 10% data from each attack traffic
df_combo_g_raw = gafgyt[0].fillna(0)
df_combo_g_raw = df_combo_g_raw[: round(0.01 * len(df_combo_g_raw))]
df_junk_g_raw = gafgyt[1].fillna(0)
df_junk_g_raw = df_junk_g_raw[: round(0.01 * len(df_junk_g_raw))]
df_scan_g_raw = gafgyt[2].fillna(0)
df_scan_g_raw = df_scan_g_raw[: round(0.01 * len(df_scan_g_raw))]


# from mirai attacks
os.chdir(mirai_path)
mirai_list = os.listdir()
mirai = []
for i in range(len(mirai_list)):
    mirai.append(pd.read_csv(mirai_list[i]))

# separate each csv file and replace nan with 0, if nan exists
df_ack_m_raw = mirai[0].fillna(0)
df_ack_m_raw = df_ack_m_raw[: round(0.01 * len(df_ack_m_raw))]
df_scan_m_raw = mirai[1].fillna(0)
df_scan_m_raw = df_scan_m_raw[: round(0.01 * len(df_scan_m_raw))]
df_syn_m_raw = mirai[2].fillna(0)
df_syn_m_raw = df_syn_m_raw[: round(0.01 * len(df_syn_m_raw))]
df_udp_m_raw = mirai[3].fillna(0)
df_udp_m_raw = df_udp_m_raw[: round(0.01 * len(df_udp_m_raw))]
df_udpplain_m_raw = mirai[4].fillna(0)
df_udpplain_m_raw = df_udpplain_m_raw[: round(0.01 * len(df_udpplain_m_raw))]

df_malicious_raw = pd.concat([df_combo_g_raw,
                              df_junk_g_raw,
                              df_scan_g_raw,
                              df_ack_m_raw,
                              df_scan_m_raw,
                              df_syn_m_raw,
                              df_udp_m_raw,
                              df_udpplain_m_raw])
"""
df_malicious_raw = pd.concat([df_combo_g_raw,
                              df_junk_g_raw,
                              df_scan_g_raw])
"""

# concatenate into training and test set
# raw data
# train_set_raw = df_train_raw
test_set_raw = df_malicious_raw

print('testset shape = {}'.format(test_set_raw.shape))
os.chdir(saved_path)
test_set_raw.to_csv('Samsung_SNH_Webcam_test_raw.csv', index=False)

# train_set_raw.to_csv('SimpleHome_1003_Security_Camera_train_raw.csv', index=False)
















