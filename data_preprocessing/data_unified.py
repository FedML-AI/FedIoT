# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 22:32:55 2021

@author: Mth
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import logging

cwd = os.getcwd()
#train_path = os.path.join(cwd, 'data_debug/train')
test_path = os.path.join(cwd, 'data_debug/1_pc/test')
saved_path = os.path.join(cwd, 'data_debug/1_pc/unified')

os.chdir(test_path)
uni_list = os.listdir()
# print(uni_list)

Uni = []
for i in range(len(uni_list)):
    Uni.append(pd.read_csv(uni_list[i]))
print('dataset shape')
for i in range(9):
    print('{} : {}'.format(uni_list[i], len(Uni[i])))
#print('## Debug mode')
#for i in range(9):
#    Uni[i] = Uni[i][: round(0.1 * len(Uni[i]))]
#    print('{} : {}'.format(uni_list[i], len(Uni[i])))


Dan = Uni[0].fillna(0)
Eco = Uni[1].fillna(0)
Enn = Uni[2].fillna(0)
Phi = Uni[3].fillna(0)
Pro_7 = Uni[4].fillna(0)
Pro_8 = Uni[5].fillna(0)
Sam = Uni[6].fillna(0)
Sim_2 = Uni[7].fillna(0)
Sim_3 = Uni[8].fillna(0)

uni_set = pd.concat([Dan, Eco, Enn, Phi, Pro_7, Pro_8, Sam, Sim_2, Sim_3])
print('Unified dataset shape is : {}'.format(len(uni_set)))

os.chdir(saved_path)
uni_set.to_csv('test_unified_1_b.csv', index=False)
