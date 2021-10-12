import argparse
import logging
import os
import sys
import random
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# import wandb
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
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=10, metavar='EP',
                        help='how many epochs will be trained')

    args = parser.parse_args()
    return args


def load_data(args):
    device_list = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
                   'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam',
                   'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']

    min = np.loadtxt('min.txt')
    max = np.loadtxt('max.txt')

    benign_train_list = list()
    benign_test_list = list()
    benign_th_list = list()
    attack_test_list = list()

    for i, device in enumerate(device_list):
        benign_data = pd.read_csv(os.path.join(args.data_dir, device, 'benign_traffic.csv'))
        benign_data = np.array(benign_data)

        benign_train = benign_data[:5000]
        benign_train[np.isnan(benign_train)] = 0
        benign_train = (benign_train - min) / (max - min)

        benign_th = benign_data[5000:8000]
        benign_th[np.isnan(benign_th)] = 0
        benign_th = (benign_th - min) / (max - min)

        benign_test = benign_data[-5000:]
        benign_test[np.isnan(benign_test)] = 0
        benign_test = (benign_test - min) / (max - min)

        g_attack_data_list = [os.path.join(args.data_dir, device, 'gafgyt_attacks', f)
                              for f in os.listdir(os.path.join(args.data_dir, device, 'gafgyt_attacks'))]
        if device == 'Ennio_Doorbell' or device == 'Samsung_SNH_1011_N_Webcam':
            attack_data_list = g_attack_data_list
            benign_test = benign_test[-2500:]
        else:
            m_attack_data_list = [os.path.join(args.data_dir, device, 'mirai_attacks', f)
                                  for f in os.listdir(os.path.join(args.data_dir, device, 'mirai_attacks'))]
            attack_data_list = g_attack_data_list + m_attack_data_list

        attack_test = pd.concat([pd.read_csv(f)[-500:] for f in attack_data_list])
        attack_test = np.array(attack_test)
        attack_test[np.isnan(attack_test)] = 0
        attack_test = (attack_test - min) / (max - min)

        benign_train_list.append(benign_train)
        benign_th_list.append(benign_th)
        benign_test_list.append(benign_test)
        attack_test_list.append(attack_test)

    
    # output = open('benign_train.pkl', 'wb')
    # pickle.dump(np.concatenate(benign_train_list), output)
    # output.close()
    # output = open('benign_test.pkl', 'wb')
    # pickle.dump(np.concatenate(benign_test_list), output)
    # output.close()
    # output = open('benign_th.pkl', 'wb')
    # pickle.dump(np.concatenate(benign_th_list), output)
    # output.close()
    # output = open('attack_test.pkl', 'wb')
    # pickle.dump(np.concatenate(attack_test_list), output)
    # output.close()

    benign_train_loader = torch.utils.data.DataLoader(np.concatenate(benign_train_list), batch_size=args.batch_size, shuffle=False, num_workers=0)
    benign_test_loader = torch.utils.data.DataLoader(np.concatenate(benign_test_list), batch_size=args.batch_size, shuffle=False, num_workers=0)
    benign_th_loader = torch.utils.data.DataLoader(np.concatenate(benign_th_list), batch_size=args.batch_size, shuffle=False, num_workers=0)
    attack_test_loader = torch.utils.data.DataLoader(np.concatenate(attack_test_list), batch_size=args.batch_size, shuffle=False, num_workers=0)

    return benign_train_loader, benign_test_loader, benign_th_loader, attack_test_loader


def load_from_pkl(args):
    pkl_file = open('benign_train.pkl', 'rb')
    benign_train = pickle.load(pkl_file)
    benign_train_loader = torch.utils.data.DataLoader(benign_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pkl_file.close()
    pkl_file = open('benign_test.pkl', 'rb')
    benign_test = pickle.load(pkl_file)
    benign_test_loader = torch.utils.data.DataLoader(benign_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pkl_file.close()
    pkl_file = open('benign_th.pkl', 'rb')
    benign_th = pickle.load(pkl_file)
    benign_th_loader = torch.utils.data.DataLoader(benign_th, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pkl_file.close()
    pkl_file = open('attack_test.pkl', 'rb')
    attack_test = pickle.load(pkl_file)
    attack_test_loader = torch.utils.data.DataLoader(attack_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pkl_file.close()

    return benign_train_loader, benign_test_loader, benign_th_loader, attack_test_loader

def create_model(args):
    model = AutoEncoder()
    logging.info(model)
    return model


def train(args, model, device, trainloader, optloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.001)
    loss_func = nn.MSELoss()
    #model training
    for epoch in range(args.epochs):
        # mini- batch loop
        epoch_loss = 0.0
        for idx, inp in enumerate(trainloader):
            optimizer.zero_grad()
            inp = inp.to(device)
            decode = model(inp)
            loss = loss_func(decode, inp)
            epoch_loss += loss.item() / args.batch_size
            loss.backward()
            optimizer.step()
        scheduler.step()
        logging.info("epoch = %d, epoch_loss = %f" % (epoch, epoch_loss))
        
    #threshold selecting
    mse = list()
    model.eval()
    thres_func = nn.MSELoss(reduction='none')
    for idx, inp in enumerate(optloader):
        mse_tr = thres_func(model(inp), inp)
        mse.append(mse_tr)

    threshold = torch.cat(mse).mean(dim=1)
    # logging.info("threshold = %d" % threshold)
    return threshold


def test(args, model, device, benignloader, anloader, threshold):
    model.eval()
    true_negative = 0
    false_positive = 0
    true_positive = 0
    false_negative = 0

    thres_func = nn.MSELoss(reduction='none')
    for idx, inp in enumerate(benignloader):
        inp = inp.to(device)
        diff = thres_func(model(inp), inp)
        mse = diff.mean(dim=1)
        false_positive += (mse > threshold).sum()
        true_negative += (mse <= threshold).sum()

    for idx, inp in enumerate(anloader):
        inp = inp.to(device)
        diff = thres_func(model(inp), inp)
        mse = diff.mean(dim=1)
        true_positive += (mse > threshold).sum()
        false_negative += (mse <= threshold).sum()

    accuracy = ((true_positive) + (true_negative)) \
                / ((true_positive) + (true_negative) + (false_positive) + (false_negative))
    precision = (true_positive) / ((true_positive) + (false_positive))
    false_positive_rate = (false_positive) / ((false_positive) + (true_negative))
    tpr = (true_positive) / ((true_positive) + (false_negative))
    tnr = (true_negative) / ((true_negative) + (false_positive))

    print('The True negative number is ', (true_negative))
    print('The False positive number is ', (false_positive))
    print('The True positive number is ', (true_positive))
    print('The False negative number is ', (false_negative))

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

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_tensor_type(torch.DoubleTensor)

    # experimental result tracking
#     wandb.init(project='fediot', entity='automl', config=args)

    # PyTorch configuration
    torch.set_default_tensor_type(torch.DoubleTensor)

    # GPU/CPU device management
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load data
    benign_train_loader, benign_test_loader, benign_th_loader, attack_test_loader = load_from_pkl(args)
    
    # create model
    model = create_model(args)
    model.to(device)

    # start training
    mse_results_global = train(args, model, device, benign_train_loader, benign_th_loader)
    # logging.info("threshold = %f" % threshold)
#     wandb.log({"Threshold": threshold})

    # start test
    threshold_global = torch.mean(mse_results_global) + 0 * torch.std(mse_results_global) 
    test(args, model, device, benign_test_loader, attack_test_loader, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 1 * torch.std(mse_results_global)
    test(args, model, device, benign_test_loader, attack_test_loader, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 2 * torch.std(mse_results_global)
    test(args, model, device, benign_test_loader, attack_test_loader, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 3 * torch.std(mse_results_global)
    test(args, model, device, benign_test_loader, attack_test_loader, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 4 * torch.std(mse_results_global)
    test(args, model, device, benign_test_loader, attack_test_loader, threshold_global)
    print(threshold_global)

    threshold_global = torch.mean(mse_results_global) + 5 * torch.std(mse_results_global)
    test(args, model, device, benign_test_loader, attack_test_loader, threshold_global)
    print(threshold_global)
    # accuracy, precision, false_positive_rate = test(args, model, device, bnloader, anloader, threshold)
