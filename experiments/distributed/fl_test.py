import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import wandb
import joblib
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from model.ae import AutoEncoder

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # dataset related
    parser.add_argument('--dataset', type=str, default='UCI_MLR', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../data/UCI-MLR',
                        help='data directory')

    # CPU/GPU device related
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu; gpu')

    # model related
    parser.add_argument('--model', type=str, default='vae',
                        help='model (default: vae): ae, vae')

    # optimizer related
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=50, metavar='EP',
                        help='how many epochs will be trained')

    args = parser.parse_args()
    return args

# for global test
def load_data(args):
    device_list = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam',
                   'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    th_local_dict = dict()
    min = np.loadtxt('min.txt')
    max = np.loadtxt('max.txt')
    for i, device in enumerate(device_list):
        benign_data = pd.read_csv(os.path.join(args.data_dir, device, 'benign_traffic.csv'))
        benign_data = np.array(benign_data)
        benign_test = benign_data[-5000:]
        benign_test[np.isnan(benign_test)] = 0
        benign_test = (benign_test - min) / (max - min)

        benign_th = benign_data[5000:8000]
        benign_th[np.isnan(benign_th)] = 0
        benign_th = (benign_th - min) / (max - min)
        
        g_attack_data_list = [os.path.join(args.data_dir, device, 'gafgyt_attacks', f)
                              for f in os.listdir(os.path.join(args.data_dir, device, 'gafgyt_attacks'))]
        if device == 'Ennio_Doorbell' or device == 'Samsung_SNH_1011_N_Webcam':
            attack_data_list = g_attack_data_list
            benign_test = benign_test[-2500:]
        else:
            m_attack_data_list = [os.path.join(args.data_dir, device, 'mirai_attacks', f)
                                  for f in os.listdir(os.path.join(args.data_dir, device, 'mirai_attacks'))]
            attack_data_list = g_attack_data_list + m_attack_data_list

        attack_data = pd.concat([pd.read_csv(f)[-500:] for f in attack_data_list])
        attack_data = np.array(attack_data)
        attack_data[np.isnan(attack_data)] = 0
        attack_data = (attack_data - min) / (max - min)

        train_data_local_dict[i] = torch.utils.data.DataLoader(benign_test, batch_size=1, shuffle=False, num_workers=0)
        test_data_local_dict[i] = torch.utils.data.DataLoader(attack_data, batch_size=1, shuffle=False, num_workers=0)
        th_local_dict[i] = torch.utils.data.DataLoader(benign_th, batch_size=1, shuffle=False, num_workers=0)

    return train_data_local_dict, test_data_local_dict, th_local_dict


def create_model(args):
    model = AutoEncoder()
    model_save_dir = "../../training"
    path = os.path.join(model_save_dir, 'model.ckpt')
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    return model

def draw(args, model, device, train_data_local_dict, test_data_local_dict):
    model.eval()
    mse_benign = list()
    mse_attack = list()
    thres_func = nn.MSELoss()

    for client_index in train_data_local_dict.keys():
        train_data = train_data_local_dict[client_index]
        for idx, inp in enumerate(train_data):
            # if idx >= round(len(train_data) * 2 / 3):
            inp = inp.to(device)
            diff = thres_func(model(inp), inp)
            mse = diff.item()
            mse_benign.append(mse)

    for client_index in test_data_local_dict.keys():
        test_data = test_data_local_dict[client_index]
        for idx, inp in enumerate(test_data):
            inp = inp.to(device)
            diff = thres_func(model(inp), inp)
            mse = diff.item()
            mse_attack.append(mse)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(range(len(mse_benign)),mse_benign)
    ax.bar(range(len(mse_attack)),mse_attack)
    plt.savefig("out.png")


def test(args, model, device, train_data_local_dict, test_data_local_dict, threshold):
    model.eval()
    true_negative = []
    false_positive = []
    true_positive = []
    false_negative = []

    thres_func = nn.MSELoss()

    for client_index in train_data_local_dict.keys():
        train_data = train_data_local_dict[client_index]
        for idx, inp in enumerate(train_data):
            # if idx >= round(len(train_data) * 2 / 3):
                inp = inp.to(device)
                diff = thres_func(model(inp), inp)
                mse = diff.item()
                if mse > threshold:
                    false_positive.append(idx)
                else:
                    true_negative.append(idx)

    for client_index in test_data_local_dict.keys():
        test_data = test_data_local_dict[client_index]
        for idx, inp in enumerate(test_data):
            inp = inp.to(device)
            diff = thres_func(model(inp), inp)
            mse = diff.item()
            if mse > threshold:
                true_positive.append(idx)
            else:
                false_negative.append(idx)


    accuracy = (len(true_positive) + len(true_negative)) \
                / (len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative))
    precision = len(true_positive) / (len(true_positive) + len(false_positive))
    false_positive_rate = len(false_positive) / (len(false_positive) + len(true_negative))
    tpr = len(true_positive) / (len(true_positive) + len(false_negative))
    tnr = len(true_negative) / (len(true_negative) + len(false_positive))

    print('The True negative number is ', len(true_negative))
    print('The False positive number is ', len(false_positive))
    print('The True positive number is ', len(true_positive))
    print('The False negative number is ', len(false_negative))

    print('The accuracy is ', accuracy)
    print('The precision is ', precision)
    print('The false positive rate is ', false_positive_rate)
    print('tpr is ', tpr)
    print('tnr is ', tnr)

    return accuracy, precision, false_positive_rate

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO,

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # experimental result tracking
    # wandb.init(project='fediot', entity='automl', config=args)

    # PyTorch configuration
    torch.set_default_tensor_type(torch.DoubleTensor)

    # GPU/CPU device management
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load data
    train_data_local_dict, test_data_local_dict, th_local_dict = load_data(args)

    # create model
    model = create_model(args)

    mse = list()
    thres_func = nn.MSELoss()
    for client_index in th_local_dict.keys():
        train_data = th_local_dict[client_index]
        for idx, inp in enumerate(train_data):
            inp = inp.to(device)
            diff = thres_func(model(inp), inp)
            mse.append(diff.item())
    mse_results_global = torch.tensor(mse)
    threshold_global = torch.mean(mse_results_global) + 0 * torch.std(mse_results_global) / np.sqrt(args.batch_size)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 1 * torch.std(mse_results_global) / np.sqrt(args.batch_size)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 2 * torch.std(mse_results_global) / np.sqrt(args.batch_size)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)

    mse = list()
    for client_index in test_data_local_dict.keys():
        test_data = test_data_local_dict[client_index]
        for idx, inp in enumerate(test_data):
            inp = inp.to(device)
            diff = thres_func(model(inp), inp)
            mse.append(diff.item())
    mse.sort()
    # threshold_global = torch.tensor(mse[round(len(mse)*0.05)])
    threshold_global = min(mse)
    test(args, model, device, train_data_local_dict, test_data_local_dict, threshold_global)
    print(threshold_global)
