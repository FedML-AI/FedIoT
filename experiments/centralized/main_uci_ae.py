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
    path_benin_traffic = args.data_dir + '/Danmini_Doorbell/Danmini_Doorbell_train_raw.csv'
    path_ack_traffic = args.data_dir + '/Danmini_Doorbell/Danmini_Doorbell_test_raw.csv'
    logging.info(path_benin_traffic)
    logging.info(path_ack_traffic)

    db_benigh = pd.read_csv(path_benin_traffic)
    db_attack = pd.read_csv(path_ack_traffic)
    trainset = db_benigh
    testset = db_attack
    trainset = (trainset - trainset.mean()) / (trainset.std())
    testset = (testset - testset.mean()) / (testset.std())
    trainset = np.array(trainset)
    testset = np.array(testset)
    testset[np.isnan(testset)] = 0
    testratio = 1 - 16515 / len(testset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return trainloader, testloader, len(trainset), len(testset), testratio
    # trainset = db_benigh[0:round(len(db_benigh) * 0.8)]
    # test_benigh = db_benigh[round(len(db_benigh) * 0.9):len(db_benigh)]
    # test_attack = db_attack[round(len(db_attack) * 0.1):round(len(db_attack) * 0.2)]
    # testset = pd.concat([test_benigh, test_attack])
    # trainset = (trainset - trainset.mean()) / (trainset.std())
    # testset = (testset - testset.mean()) / (testset.std())
    # trainset = np.array(trainset)
    # testset = np.array(testset)
    # testset = torch.Tensor(testset)
    # trainset = torch.Tensor(trainset)
    # testratio = 1 - abs((round(len(db_benigh) * 0.9) - len(db_benigh)) / len(testset))
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    # return trainloader, testloader, len(trainset), len(testset), testratio


def create_model(args):
    model = AutoEncoder()
    logging.info(model)
    return model


def train(args, model, device, trainloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()
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

    i = []
    model.eval()
    for idx, inp in enumerate(trainloader):
        i.append(torch.sum(torch.square(inp - model(inp)))/115/args.batch_size)
    i.sort()
    len_i = len(i)
    i = i[1:round(len_i * 0.95)]
    i = torch.tensor(i)
    # test = np.array(i)
    # plt.hist(test, bins='auto', density=True)
    # plt.show()
    threshold = (torch.mean(i) + 2 * torch.std(i))
    logging.info("threshold = %d" % threshold)
    # test = np.array(i)
    # plt.hist(test, bins='auto', density=True)
    # plt.show()
    return threshold


def test(args, model, device, testloader, test_len, testratio):
    model.eval()
    anomaly = []
    for idx, inp in enumerate(testloader):
        inp = inp.to(device)
        diff = torch.sum(torch.square(inp - model(inp)))/115
        if idx > 16515:
            logging.info("idx = %d, mse = %f" % (idx, diff))
        if diff > threshold:
            anomaly.append(idx)
    precision = (len(anomaly)/test_len)/testratio
    print('The accuracy is ', precision)
    print('The length of the test set is ', test_len)
    print('The number of the detected anomaly is ', len(anomaly))
    wandb.log({"Precision": precision})
    return precision


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
    trainloader, testloader, train_len, test_len, test_ratio = load_data(args)

    # create model
    model = create_model(args)
    model.to(device)

    # start training
    threshold = train(args, model, device, trainloader)
    logging.info("threshold = %f" % threshold)

    # start test
    precision = test(args, model, device, testloader, test_len, test_ratio)
