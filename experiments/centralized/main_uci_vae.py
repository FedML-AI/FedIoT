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

from model.vae import VAE


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
    path_benin_traffic = args.data_dir + '/benign_traffic.csv'
    path_ack_traffic = args.data_dir + '/ack.csv'
    logging.info(path_benin_traffic)
    logging.info(path_ack_traffic)

    db_benigh = pd.read_csv(path_benin_traffic)
    db_attack = pd.read_csv(path_ack_traffic)
    # trainset = torch.Tensor(db_benigh)
    # testset = torch.Tensor(db_attack[0:round(len(db_attack) * 0.2)])
    # trainset = (trainset - trainset.mean()) / (trainset.std())
    # testset = (testset - testset.mean()) / (testset.std())
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    # return trainloader, testloader, len(trainset), len(testset)
    trainset = db_benigh[0:round(len(db_benigh) * 0.8)]
    test_benigh = db_benigh[round(len(db_benigh) * 0.9):len(db_benigh)]
    test_attack = db_attack[round(len(db_attack) * 0.1):round(len(db_attack) * 0.2)]
    testset = pd.concat([test_benigh, test_attack])
    trainset = (trainset - trainset.mean()) / (trainset.std())
    testset = (testset - testset.mean()) / (testset.std())
    trainset = np.array(trainset)
    testset = np.array(testset)
    testset = torch.Tensor(testset)
    trainset = torch.Tensor(trainset)
    testratio = 1 - abs((round(len(db_benigh) * 0.9) - len(db_benigh)) / len(testset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return trainloader, testloader, len(trainset), len(testset), testratio


def create_model(device):
    model = VAE(device)
    logging.info(model)
    return model


def loss_function_vae(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    reconstruction_function = nn.BCELoss(size_average=False)  # mse loss
    BCE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


def train(args, model, device, trainloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        # mini- batch loop
        epoch_loss = 0.0
        for idx, inp in enumerate(trainloader):
            inp = inp.to(device)
            optimizer.zero_grad()
            decode, mu, logvar = model(inp)
            loss = loss_function_vae(decode, inp, mu, logvar)
            epoch_loss += loss.item() / args.batch_size
            loss.backward()
            optimizer.step()
        logging.info("epoch = %d, epoch_loss = %f" % (epoch, epoch_loss))
        wandb.log({"loss": epoch_loss, "epoch": epoch})
    logging.info("batch size = %d" % args.batch_size)

    i = []
    model.eval()
    for idx, inp in enumerate(trainloader):
        inp = inp.to(device)
        decode, mu, logvar = model(inp)
        i.append(torch.sum(abs(decode - inp)))
    i = torch.tensor(i)
    test = np.array(i)
    plt.hist(test, bins='auto', density=True)
    plt.show()
    threshold = (torch.mean(i) + 0.8 * torch.std(i)) / args.batch_size
    test = np.array(i)
    plt.hist(test, bins='auto', density=True)
    plt.show()
    wandb.log({"threshold": threshold})
    return threshold


def test(args, model, device, testloader, test_len, testratio):
    anmoaly = []
    for idx, inp in enumerate(testloader):
        inp = inp.to(device)
        decode, mu, logvar = model(inp)
        diff = torch.sum(abs(inp - decode))
        if diff > threshold:
            anmoaly.append(idx)
    an_ratio = (len(anmoaly) / test_len)
    precision = 1 - (abs(an_ratio - testratio) / testratio)
    print('The accuracy is ', precision)
    len(trainloader.dataset)

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

    # PyTorch Configuration
    torch.set_default_tensor_type(torch.DoubleTensor)

    # experimental result tracking
    wandb.init(project='fediot', entity='automl', config=args)

    # GPU/CPU device management
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # load data
    trainloader, testloader, train_len, test_len, test_ratio = load_data(args)

    # create model
    model = create_model(device)
    model.to(device)

    # start training
    threshold = train(args, model, device, trainloader)
    logging.info("threshold = %f" % threshold)

    # start test
    precision = test(args, model, device, testloader, test_len, test_ratio)
