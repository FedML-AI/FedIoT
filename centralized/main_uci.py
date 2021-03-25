import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from centralized.auto_encoder import AutoEncoder


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='vae', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='UCI_MLR', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../data/UCI-MLR',
                        help='data directory')

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
    path_benin_traffic = args.data_dir + '/benign_traffic.csv'
    path_ack_traffic = args.data_dir + '/ack.csv'
    logging.info(path_benin_traffic)
    logging.info(path_ack_traffic)

    db_benigh = np.array(pd.read_csv(path_benin_traffic))
    db_attack = np.array(pd.read_csv(path_ack_traffic))
    trainset = torch.Tensor(db_benigh)
    testset = torch.Tensor(db_attack[0:round(len(db_attack) * 0.1)])
    trainset = (trainset - trainset.mean()) / (trainset.std())
    testset = (testset - testset.mean()) / (testset.std())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return trainloader, testloader, len(trainset), len(testset)


def train(args, trainloader):
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    loss_func = nn.MSELoss()
    running_loss = 0.0
    for epoch in range(50):

        # mini- batch loop
        epoch_loss = 0.0
        for idx, inp in enumerate(trainloader):
            optimizer.zero_grad()
            decode = autoencoder(inp)
            loss = loss_func(decode, inp)
            epoch_loss += loss.item() / args.batch_size
            loss.backward()
            optimizer.step()
        logging.info("epoch = %d, epoch_loss = %f" % (epoch, epoch_loss))
        wandb.log({"loss": epoch_loss, "epoch": epoch})
        running_loss += epoch_loss

    threshold = 0
    a = 0
    for idx, inp in enumerate(trainloader):
        i = max(sum(abs(autoencoder(inp) - inp)))
        logging.info("inp is: " + str(inp))
        logging.info("output is: " + str(autoencoder(inp)))
        logging.info("number is: " + str(idx))
        if i > threshold:
            threshold = i
        a += 1
    logging.info('threshold is = %f' % threshold)
    wandb.summary
    return threshold


def test(args, testloader, test_len):
    autoencoder.eval()
    anmoaly = []
    for idx, inp in enumerate(testloader):
        decode = autoencoder(inp)
        diff = torch.sum(abs(inp - decode))
        if diff > 0.3516:
            anmoaly.append(idx)
    an_ratio = len(anmoaly) / test_len
    print('The accuracy is ', an_ratio)

    len(trainloader.dataset)

    torch.save(autoencoder.state_dict(), "vae_v1.pth")
    print("Saved PyTorch Model State to model.pth")


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

    # load data
    trainloader, testloader, train_len, test_len = load_data(args)

    # create model
    autoencoder = AutoEncoder()
    logging.info(autoencoder)

    # start training
    threshold = train(args, trainloader)
    logging.info("threshold = %f" % threshold)

    # start test
    test(args, testloader, test_len)
