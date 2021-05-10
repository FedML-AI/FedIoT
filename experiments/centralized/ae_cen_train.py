import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
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
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
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


def load_data(args):
    path_benin_traffic = args.data_dir + '/new_centralized_set/centralized_unified_bgh.csv'
    logging.info(path_benin_traffic)

    db_benign = pd.read_csv(path_benin_traffic)
    db_benign = (db_benign - db_benign.mean()) / (db_benign.std())
    db_benign[np.isnan(db_benign)] = 0
    trainset = db_benign[0:round(len(db_benign) * 0.67)]
    optset = db_benign[round(len(db_benign) * 0.67):len(db_benign)]
    trainset = np.array(trainset)
    optset = np.array(optset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optloader = torch.utils.data.DataLoader(optset, batch_size= args.batch_size, shuffle=False, num_workers=0)
    logging.info('train length is %d' %(len(trainset)))
    return trainloader, optloader, len(trainset), len(optset)


def create_model(args):
    model = AutoEncoder()
    logging.info(model)
    return model


def train(args, model, device, trainloader, optloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        logging.info("epoch = %d, epoch_loss = %f" % (epoch, epoch_loss))
        wandb.log({"loss": epoch_loss, "epoch": epoch})
    logging.info("batch size = %d" % args.batch_size)

    # #threshold selecting
    # i = []
    # model.eval()
    # thres_func = nn.MSELoss()
    # for idx, inp in enumerate(optloader):
    #     mse_tr = thres_func(model(inp), inp)
    #     i.append(mse_tr.item())
    # i.sort()
    # len_i = len(i)
    # i = i[round(len_i * 0.00):round(len_i * 1)]
    # i = torch.tensor(i)
    # # test = np.array(i)
    # # plt.hist(test, bins='auto', density=True)
    # # plt.show()
    # threshold = (torch.mean(i) + 1 * torch.std(i) / np.sqrt(args.batch_size))
    # logging.info("threshold = %d" % threshold)
    # return threshold

def test(args, model, device, test_index):

    if test_index == 0:
        path_benin_traffic = args.data_dir + '/Danmini_Doorbell/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/Danmini_Doorbell_test_raw.csv'
    if test_index == 1:
        path_benin_traffic = args.data_dir + '/Ecobee_Thermostat/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/Ecobee_Thermostat_test_raw.csv'
    if test_index == 2:
        path_benin_traffic = args.data_dir + '/Ennio_Doorbell/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/Ennio_Doorbell_test_raw.csv'
    if test_index == 3:
        path_benin_traffic = args.data_dir + '/Philips_B120N10_Baby_Monitor/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/Philips_Baby_Monitor_test_raw.csv'
    if test_index == 4:
        path_benin_traffic = args.data_dir + '/Provision_PT_737E_Security_Camera/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/Provision_737E_Security_Camera_test_raw.csv'
    if test_index == 5:
        path_benin_traffic = args.data_dir + '/Provision_PT_838_Security_Camera/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/Provision_838_Security_Camera_test_raw.csv'
    if test_index == 6:
        path_benin_traffic = args.data_dir + '/Samsung_SNH_1011_N_Webcam/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/Samsung_SNH_Webcam_test_raw.csv'
    if test_index == 7:
        path_benin_traffic = args.data_dir + '/SimpleHome_XCS7_1002_WHT_Security_Camera/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/SimpleHome_1002_Security_Camera_test_raw.csv'
    if test_index == 8:
        path_benin_traffic = args.data_dir + '/SimpleHome_XCS7_1003_WHT_Security_Camera/benign_traffic.csv'
        path_ack_traffic = args.data_dir + '/new_centralized_set/Atk_data/SimpleHome_1003_Security_Camera_test_raw.csv'

    db_benign = pd.read_csv(path_benin_traffic)
    db_attack = pd.read_csv(path_ack_traffic)
    db_benign = (db_benign - db_benign.mean()) / (db_benign.std())
    db_attack = (db_attack - db_attack.mean()) / (db_attack.std())
    db_benign[np.isnan(db_benign)] = 0
    db_attack[np.isnan(db_attack)] = 0
    testset = pd.concat([db_benign[round(len(db_benign) * 0.67):len(db_benign)], db_attack])
    test_tr = len(db_benign) - round(len(db_benign) * 0.67)
    testset = np.array(testset)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    logging.info('testloader is finished')

    optset = db_benign[round(len(db_benign) * 0.67):len(db_benign)]
    optset = np.array(optset)
    optloader = torch.utils.data.DataLoader(optset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logging.info('optloader is finished')

    model.eval()

    i = []
    model.eval()
    thres_func = nn.MSELoss()
    for idx, inp in enumerate(optloader):
        mse_tr = thres_func(model(inp), inp)
        i.append(mse_tr.item())
    i.sort()
    len_i = len(i)
    i = i[round(len_i * 0.00):round(len_i * 1)]
    i = torch.tensor(i)
    threshold = (torch.mean(i) + 1 * torch.std(i) / np.sqrt(args.batch_size))
    logging.info("threshold = %d" % threshold)


    true_negative = []
    false_positive = []
    true_positive = []
    false_negative = []

    thres_func = nn.MSELoss()
    for idx, inp in enumerate(testloader):
        inp = inp.to(device)
        diff = thres_func(model(inp), inp)
        mse = diff.item()
        # logging.info("idx is %d, mse is %f" %(idx, mse))
        if idx <= test_tr:
            if mse > threshold:
                false_positive.append(idx)
            else:
                true_negative.append(idx)
        else:
            if mse > threshold:
                true_positive.append(idx)
            else:
                false_negative.append(idx)

    accuracy = (len(true_positive) + len(true_negative)) \
                / (len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative))
    precision = len(true_positive) / (len(true_positive) + len(false_positive))
    false_positive_rate = len(false_positive) / (len(false_positive) + len(true_negative))

    print('Device index is ', test_index)
    print('The Threshold is ', threshold)
    print('The True negative number is ', len(true_negative))
    print('The False positive number is ', len(false_positive))
    print('The True positive number is ', len(true_positive))
    print('The False negative number is ', len(false_negative))

    print('The accuracy is ', accuracy)
    print('The precision is ', precision)
    print('The false positive rate is ', false_positive_rate)

    wandb.log({"accuracy": accuracy})
    wandb.log({"precision": precision})
    wandb.log({"false positive rate": false_positive_rate})

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
    wandb.init(project='fediot', entity='automl', config=args)

    # PyTorch configuration
    torch.set_default_tensor_type(torch.DoubleTensor)

    # GPU/CPU device management
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load data
    trainloader, optloader, train_len, opt_len = load_data(args)

    # create model
    model = create_model(args)
    model.to(device)

    # start training
    train(args, model, device, trainloader, optloader)
    # logging.info("threshold = %f" % threshold)
    # wandb.log({"Threshold": threshold})

    # start test
    for i in range(9):
        accuracy, precision, false_positive_rate = test(args, model, device, i)