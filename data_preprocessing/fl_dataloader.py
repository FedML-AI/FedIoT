import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

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

    parser.add_argument('--client_num_in_total', type=int, default=9, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
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


# read the data from the csv
def load_data(args, benign_file_name, attack_file_name):
    path_benin_traffic = args.data_dir + "/" + benign_file_name
    path_ack_traffic = args.data_dir + "/" + attack_file_name
    logging.info(path_benin_traffic)
    logging.info(path_ack_traffic)

    db_benign = pd.read_csv(path_benin_traffic)
    db_attack = pd.read_csv(path_ack_traffic)
    db_benign = (db_benign - db_benign.mean()) / (db_benign.std())
    # db_attack = (db_attack - db_attack.mean()) / (db_attack.std())

    trainset = np.array(db_benign)
    testset = np.array(db_attack)
    trainset[np.isnan(trainset)] = 0
    # testset[np.isnan(testset)] = 0
    testset = testset[9000:15199]

    len_train = len(trainset)
    len_test = len(testset)
    logging.info('Origin csv data is loaded, waiting for partition')
    return len_train, len_test, trainset, testset


def homo_partition_data(args, process_id, benign_data, attack_data):
    device_num = 9
    if process_id == 0:  # for centralized training
        return benign_data, attack_data
    else:
        # For benign set
        ## records the range of idxs for each device
        benign_split_list = [0, 49548, 62661, 101761, 277001, 339155, 437669, 489819, 536404, 555932]
        ## records the number of samples of each device
        benign_len = [49548, 13113, 39100, 175240, 62154, 98514, 52150, 46585, 19528]

        # For attack set
        ## records the range of idxs for each device
        # attack_split_list = [0, 77073, 139367, 150460, 222967, 278726, 333160, 344627, 407020, 470044]
        #attack_split_list = [0, 385366, 696841, 752306, 1114843, 1393634, 1665799, 1723134, 2035103, 2350225]
        attack_split_list = [0, 799, 1599, 1899, 2699, 3499, 4299, 4599, 5399, 6199]

        # training and opt data are from the unified benign dataset
        train_dataidx_map = {}
        opt_dataidx_map = {}
        attack_dataidx_map = {}

        # loop for 9 devices
        for i in range(device_num):
            # index range for train
            benign_idxs = list(np.arange(benign_split_list[i], benign_split_list[i + 1]))
            # separate train range and opt range
            train_dataidx_map[i] = benign_idxs
            opt_dataidx_map[i] = benign_idxs[benign_len[i] - 1000:]
            # index range for attack
            attack_idxs = list(np.arange(attack_split_list[i], attack_split_list[i + 1]))
            attack_dataidx_map[i] = attack_idxs
        # record the number of samples
        train_data_local_num_dict = {i: len(train_dataidx_map[i]) for i in range(device_num)}
        #opt_data_local_num_dict = {i: len(opt_dataidx_map[i]) for i in range(device_num)}
        attack_data_local_num_dict = {i: len(attack_dataidx_map[i]) for i in range(device_num)}

    logging.info('Partition is completed, waiting for input')
    return train_dataidx_map, train_data_local_num_dict, opt_dataidx_map,\
           attack_dataidx_map, attack_data_local_num_dict


def local_dataloader(args, benign_file_name, attack_file_name, process_id):
    # Dict: records dataloaders for each devices
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    #opt_data_local_dict = dict()

    # get training and test set
    train_data_num, test_data_num, train_data_global, test_data_global \
        = load_data(args, benign_file_name, attack_file_name)

    dataidx_map_train, train_data_local_num_dict, opt_dataidx_map,\
    dataidx_map_attack, attack_data_local_num_dict = homo_partition_data(args, process_id, \
                                                                         train_data_global, test_data_global)

    # for local train data
    for client_idx in range(args.client_num_in_total):
        data_local_train = train_data_global[dataidx_map_train[client_idx]]
        train_data_local_dict[client_idx] = torch.utils.data.DataLoader(data_local_train, batch_size=args.batch_size,
                                                                        shuffle=False,
                                                                        num_workers=0)
        logging.info(
            "client_idx = %d, local_train_sample_number = %d" % (client_idx, train_data_local_num_dict[client_idx]))

    # for local opt and test data
    for client_idx in range(args.client_num_in_total):
        data_local_attack = test_data_global[dataidx_map_attack[client_idx]]
        data_local_opt = train_data_global[opt_dataidx_map[client_idx]]
        data_local_test = np.concatenate((data_local_opt, data_local_attack), axis=0)

        test_data_local_dict[client_idx] = torch.utils.data.DataLoader(data_local_test, batch_size=1,
                                                                       shuffle=False,
                                                                       num_workers=0)

        logging.info("true local test sample number = %d, real local_test_sample_number = %d" % (len(opt_dataidx_map[client_idx]) +
                                                                                                 attack_data_local_num_dict[client_idx],\
                                                                                                 len(data_local_test)))
        # logging.info('Local test sample number = %d' % ())
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    dataset = local_dataloader(args, '/federated_learning_data/train_unified.csv',
                               '/new_centralized_set/global_testset_test.csv', 1)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict] = dataset
