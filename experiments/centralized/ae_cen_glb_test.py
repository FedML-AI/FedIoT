import argparse
import logging
import os
import sys

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
    path_benign_train = args.data_dir + '/benign_train.csv'
    path_benign_opt = args.data_dir + '/benign_opt.csv'
    path_benign_test = args.data_dir + '/benign_test.csv'
    path_attack_test = args.data_dir + '/attack_test.csv'
    logging.info(path_benign_train)
    logging.info(path_benign_opt)
    logging.info(path_benign_test)
    logging.info(path_attack_test)

    db_benign_train = pd.read_csv(path_benign_train)
    db_benign_opt = pd.read_csv(path_benign_opt)
    db_benign_test = pd.read_csv(path_benign_test)
    db_attack_test = pd.read_csv(path_attack_test)
    db_benign_train = (db_benign_train - db_benign_train.mean()) / (db_benign_train.std())
    db_benign_opt = (db_benign_opt - db_benign_opt.mean()) / (db_benign_opt.std())
    db_benign_test = (db_benign_test - db_benign_test.mean()) / (db_benign_test.std())
    db_attack_test = (db_attack_test - db_attack_test.mean()) / (db_attack_test.std())
    db_benign_train[np.isnan(db_benign_train)] = 0
    db_benign_opt[np.isnan(db_benign_opt)] = 0
    db_benign_test[np.isnan(db_benign_test)] = 0
    db_attack_test[np.isnan(db_attack_test)] = 0

    trainset = np.array(db_benign_train)
    optset = np.array(db_benign_opt)
    test_benign = np.array(db_benign_test)
    test_anmoaly = np.array(db_attack_test)

    bnloader = torch.utils.data.DataLoader(test_benign, batch_size=1, shuffle=False, num_workers=0)
    anloader = torch.utils.data.DataLoader(test_anmoaly, batch_size=1, shuffle=False, num_workers=0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optloader = torch.utils.data.DataLoader(optset, batch_size= args.batch_size, shuffle=False, num_workers=0)

    return trainloader, optloader, bnloader, anloader


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


    #threshold selecting
    i = []
    model.eval()
    thres_func = nn.MSELoss()
    for idx, inp in enumerate(optloader):
        mse_tr = thres_func(model(inp), inp)
        i.append(mse_tr.item())

    i.sort()
    len_i = len(i)
    i = torch.tensor(i)
    threshold = (torch.mean(i))
    logging.info("threshold = %d" % threshold)
    return threshold


def test(args, model, device, benignloader, anloader, threshold):
    model.eval()
    true_negative = []
    false_positive = []
    true_positive = []
    false_negative = []

    thres_func = nn.MSELoss()
    for idx, inp in enumerate(benignloader):
        inp = inp.to(device)
        diff = thres_func(model(inp), inp)
        mse = diff.item()
        if mse > threshold:
            false_positive.append(idx)
        else:
            true_negative.append(idx)

    for idx, inp in enumerate(anloader):
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

    print('The True negative number is ', len(true_negative))
    print('The False positive number is ', len(false_positive))
    print('The True positive number is ', len(true_positive))
    print('The False negative number is ', len(false_negative))

    print('The accuracy is ', accuracy)
    print('The precision is ', precision)
    print('The false positive rate is ', false_positive_rate)

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
#     wandb.init(project='fediot', entity='automl', config=args)

    # PyTorch configuration
    torch.set_default_tensor_type(torch.DoubleTensor)

    # GPU/CPU device management
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load data
    trainloader, optloader, bnloader, anloader = load_data(args)

    # create model
    model = create_model(args)
    model.to(device)

    # start training
    threshold = train(args, model, device, trainloader, optloader)
    logging.info("threshold = %f" % threshold)
#     wandb.log({"Threshold": threshold})

    # start test
    accuracy, precision, false_positive_rate = test(args, model, device, bnloader, anloader, threshold)
