import numpy as np
import pandas as pd
import os

cwd = os.getcwd()

# loading path is where the raw data is stored
load_path = os.path.join(cwd, '../data/UCI-MLR/new_centralized_set')

# saved path is to which the integrated data are stored
saved_path = os.path.join(cwd, '../data/UCI-MLR/new_centralized_set')

#read data
os.chdir(load_path)
uni_list = os.listdir()
Uni = []
for i in range(len(uni_list)):
    Uni.append(pd.read_csv(uni_list[i]))

for i in range(len(uni_list)):
    print('{} : {}'.format(uni_list[i], len(Uni[i])))


uni_set = pd.concat([Uni[0], Uni[1], Uni[2]])
print('Unified dataset shape is : {}'.format(len(uni_set)))

os.chdir(saved_path)
uni_set.to_csv('centralized_unified_bgh.csv', index=False)
print('finished')