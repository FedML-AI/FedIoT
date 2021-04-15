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
    path_benin_traffic = args.data_dir + '/Ennio_Doorbell/Ennio_Doorbell_benign_raw.csv'
    path_ack_traffic = args.data_dir + '/Ennio_Doorbell/Ennio_Doorbell_atk_raw.csv'
    logging.info(path_benin_traffic)
    logging.info(path_ack_traffic)

    db_benign = pd.read_csv(path_benin_traffic)
    db_attack = pd.read_csv(path_ack_traffic)
    db_benign = (db_benign - db_benign.mean()) / (db_benign.std())
    db_attack = (db_attack - db_attack.mean()) / (db_attack.std())
    db_benign[np.isnan(db_benign)] = 0
    db_attack[np.isnan(db_attack)] = 0
    trainset = db_benign[0:round(len(db_benign) * 0.67)]
    optset = db_benign[round(len(db_benign) * 0.67):len(db_benign)]
    testset = pd.concat([db_benign[round(len(db_benign) * 0.67):len(db_benign)], db_attack])
    trainset = np.array(trainset)
    optset = np.array(optset)
    testset = np.array(testset)
    testratio = 1 - len(trainset) / (2 * len(testset))
    test_tr = len(trainset) * 0.5
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optloader = torch.utils.data.DataLoader(optset, batch_size= args.batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    logging.info('train length is %d, test_tr is %f' %(len(trainset),test_tr))
    return trainloader, testloader, optloader, len(trainset), len(testset), testratio, test_tr
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

    #threshold selecting
    i = []
    model.eval()
    thres_func = nn.MSELoss()
    for idx, inp in enumerate(optloader):
        mse_tr = thres_func(model(inp), inp)
        i.append(mse_tr.item())
        # i.append(torch.sum(torch.square(inp - model(inp))/115/args.batch_size))
        # i.append(torch.sqrt(torch.sum(torch.square(inp - model(inp))) / 115 / args.batch_size))
    i.sort()
    len_i = len(i)
    i = i[round(len_i * 0.00):round(len_i * 1)]
    i = torch.tensor(i)
    # test = np.array(i)
    # plt.hist(test, bins='auto', density=True)
    # plt.show()
    threshold = (torch.mean(i) + 1 * torch.std(i) / np.sqrt(args.batch_size))
    logging.info("threshold = %d" % threshold)
    # test = np.array(i)
    # plt.hist(test, bins='auto', density=True)
    # plt.show()
    return threshold


def test(args, model, device, testloader, test_len, testratio, test_tr):
    model.eval()

    true_negative = []
    false_positive = []
    true_positive = []
    false_negative = []

    thres_func = nn.MSELoss()
    for idx, inp in enumerate(testloader):
        inp = inp.to(device)
        diff = thres_func(model(inp), inp)
        mse = diff.item()
        logging.info("idx is %d, mse is %f" %(idx, mse))
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
        # if idx > 10000:
        #     logging.info("idx = %d, mse = %f" % (idx, diff))
        # if mse > threshold:
        #     anmoaly.append(idx)
        # else:
        #     logging.info("idx = %d, mse = %f" % (idx, diff))

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
    trainloader, testloader, optloader, train_len, test_len, test_ratio, test_tr = load_data(args)

    # create model
    model = create_model(args)
    model.to(device)

    # start training
    threshold = train(args, model, device, trainloader, optloader)
    logging.info("threshold = %f" % threshold)
    wandb.log({"Threshold": threshold})

    # start test
    accuracy, precision, false_positive_rate = test(args, model, device, testloader, test_len, test_ratio, test_tr)
