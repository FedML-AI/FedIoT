import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))


def local_dataloader(args):
    device_list = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam',
                   'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']
    train_data_global = list()
    test_data_global = list()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_num = 0
    test_data_num = 0
    for i, device in enumerate(device_list):
        benign_data = pd.read_csv(os.path.join(args.data_dir, device, 'benign_traffic.csv'))
        benign_data = (benign_data - benign_data.mean()) / (benign_data.std())
        benign_data = np.array(benign_data)
        benign_data[np.isnan(benign_data)] = 0

        g_attack_data_list = [os.path.join(args.data_dir, device, 'gafgyt_attacks', f)
                              for f in os.listdir(os.path.join(args.data_dir, device, 'gafgyt_attacks'))]
        if device == 'Ennio_Doorbell' or device == 'Samsung_SNH_1011_N_Webcam':
            attack_data_list = g_attack_data_list
        else:
            m_attack_data_list = [os.path.join(args.data_dir, device, 'mirai_attacks', f)
                                  for f in os.listdir(os.path.join(args.data_dir, device, 'mirai_attacks'))]
            attack_data_list = g_attack_data_list + m_attack_data_list

        attack_data = pd.concat([pd.read_csv(f)[:500] for f in attack_data_list])
        attack_data = (attack_data - attack_data.mean()) / (attack_data.std())
        attack_data = np.array(attack_data)
        attack_data[np.isnan(attack_data)] = 0

        train_data_local_dict[i] = torch.utils.data.DataLoader(benign_data,
                                                               batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_data_local_dict[i] = torch.utils.data.DataLoader(attack_data,
                                                              batch_size=1, shuffle=False, num_workers=0)
        train_data_local_num_dict[i] = round(len(train_data_local_dict[i]) * 2 / 3) * args.batch_size
        train_data_num += round(len(train_data_local_dict[i]) * 2 / 3) * args.batch_size
        test_data_num += len(attack_data)

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
    # logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./../data/UCI-MLR',
                        help='data directory')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    args = parser.parse_args()
    logging.info(args)

    dataset = local_dataloader(args)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict] = dataset
    print(dataset)
