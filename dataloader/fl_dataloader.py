import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

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

#read the data from the csv
def load_data(args,file_name):
    path_data = args.data_dir + "/" + file_name
    logging.info(path_data)
    db = pd.read_csv(path_data)
    db = (db - db.mean()) / (db.std())
    db = np.array(db)
    return db

def homo_partition_data(process_id, dataset, client_number):
    if process_id == 0: #for centralized training
        return dataset
    else:
        total_num = len(dataset)
        idxs = list(range(total_num))
        batch_idxs = np.array_split(idxs,client_number)
        net_dataidx_map = {i: batch_idxs[i] for i in range(client_number)}
        return net_dataidx_map

def local_dataloader(process_id, net_dataidx_map, dataset, client_number):

    if process_id == 0:
        return dataset
    else:
        data_local = {}
        for client_idx in range(client_number):
            dataidxs = net_dataidx_map[client_idx]
            local_data_num = len(dataidxs)
            logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
            data_local[client_idx] = dataset[net_dataidx_map[client_idx]]

        return data_local

if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # load data
    dataset = load_data(args,'benign_traffic.csv')

    #data partition

    local_data_map = homo_partition_data(1, dataset, 20)

    #local data dict

    local_data_dict = local_dataloader(1, local_data_map, dataset, 20)

    logging.info(local_data_dict[0])
    logging.info(len(local_data_dict[0]))


