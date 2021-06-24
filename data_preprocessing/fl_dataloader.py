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
def load_data(args):
    Danmini = pd.read_csv('/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Danmini_Doorbell/benign_traffic.csv')
    Danmini = (Danmini - Danmini.mean()) / (Danmini.std())
    Danmini = np.array(Danmini)
    Danmini[np.isnan(Danmini)] = 0

    Ecobee = pd.read_csv('/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Ecobee_Thermostat/benign_traffic.csv')
    Ecobee = (Ecobee - Ecobee.mean()) / (Ecobee.std())
    Ecobee = np.array(Ecobee)
    Ecobee[np.isnan(Ecobee)] = 0

    Ennio = pd.read_csv('/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Ennio_Doorbell/benign_traffic.csv')
    Ennio = (Ennio - Ennio.mean()) / (Ennio.std())
    Ennio = np.array(Ennio)
    Ennio[np.isnan(Ennio)] = 0

    Philips = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Philips_B120N10_Baby_Monitor/benign_traffic.csv')
    Philips = (Philips - Philips.mean()) / (Philips.std())
    Philips = np.array(Philips)
    Philips[np.isnan(Philips)] = 0

    Provision_73 = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Provision_PT_737E_Security_Camera/benign_traffic.csv')
    Provision_73 = (Provision_73 - Provision_73.mean()) / (Provision_73.std())
    Provision_73 = np.array(Provision_73)
    Provision_73[np.isnan(Provision_73)] = 0

    Provision_83 = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Provision_PT_838_Security_Camera/benign_traffic.csv')
    Provision_83 = (Provision_83 - Provision_83.mean()) / (Provision_83.std())
    Provision_83 = np.array(Provision_83)
    Provision_83[np.isnan(Provision_83)] = 0

    Samsung = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Samsung_SNH_1011_N_Webcam/benign_traffic.csv')
    Samsung = (Samsung - Samsung.mean()) / (Samsung.std())
    Samsung = np.array(Samsung)
    Samsung[np.isnan(Samsung)] = 0

    Simple_02 = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/SimpleHome_XCS7_1002_WHT_Security_Camera/benign_traffic.csv')
    Simple_02 = (Simple_02 - Simple_02.mean()) / (Simple_02.std())
    Simple_02 = np.array(Simple_02)
    Simple_02[np.isnan(Simple_02)] = 0

    Simple_03 = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/SimpleHome_XCS7_1003_WHT_Security_Camera/benign_traffic.csv')
    Simple_03 = (Simple_03 - Simple_03.mean()) / (Simple_03.std())
    Simple_03 = np.array(Simple_03)
    Simple_03[np.isnan(Simple_03)] = 0

    train_data_num = len(Danmini) + len(Ecobee) + len(Ennio) + len(Philips) + len(Provision_73) + len(
        Provision_83) + len(Samsung) + len(Simple_02) + len(Simple_03)
    train_data_global = pd.read_csv('/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/benign_traffic_unified.csv')
    train_data_global = (train_data_global - train_data_global.mean()) / (train_data_global.std())
    train_data_global = np.array(train_data_global)
    train_data_global[np.isnan(train_data_global)] = 0

    Danmini_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Danmini_Doorbell/Danmini_Doorbell_attack_test.csv')
    Danmini_test = (Danmini_test - Danmini_test.mean()) / (Danmini_test.std())
    Danmini_test = np.array(Danmini_test)
    Danmini_test[np.isnan(Danmini_test)] = 0

    Ecobee_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Ecobee_Thermostat/Ecobee_Thermostat_attack_test.csv')
    Ecobee_test = (Ecobee_test - Ecobee_test.mean()) / (Ecobee_test.std())
    Ecobee_test = np.array(Ecobee_test)
    Ecobee_test[np.isnan(Ecobee_test)] = 0

    Ennio_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Ennio_Doorbell/Ennio_Doorbell_attack_test.csv')
    Ennio_test = (Ennio_test - Ennio_test.mean()) / (Ennio_test.std())
    Ennio_test = np.array(Ennio_test)
    Ennio_test[np.isnan(Ennio_test)] = 0

    Philips_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Philips_B120N10_Baby_Monitor/Philips_B120N10_Baby_Monitor_attack_test.csv')
    Philips_test = (Philips_test - Philips_test.mean()) / (Philips_test.std())
    Philips_test = np.array(Philips_test)
    Philips_test[np.isnan(Philips_test)] = 0

    Provision_73_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Provision_PT_737E_Security_Camera/Provision_PT_737E_Security_Camera_attack_test.csv')
    Provision_73_test = (Provision_73_test - Provision_73_test.mean()) / (Provision_73_test.std())
    Provision_73_test = np.array(Provision_73_test)
    Provision_73_test[np.isnan(Provision_73_test)] = 0

    Provision_83_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Provision_PT_838_Security_Camera/Provision_PT_838_Security_Camera_attack_test.csv')
    Provision_83_test = (Provision_83_test - Provision_83_test.mean()) / (Provision_83_test.std())
    Provision_83_test = np.array(Provision_83_test)
    Provision_83_test[np.isnan(Provision_83_test)] = 0

    Samsung_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/Samsung_SNH_1011_N_Webcam/Samsung_SNH_1011_N_Webcam_attack_test.csv')
    Samsung_test = (Samsung_test - Samsung_test.mean()) / (Samsung_test.std())
    Samsung_test = np.array(Samsung_test)
    Samsung_test[np.isnan(Samsung_test)] = 0

    Simple_02_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/SimpleHome_XCS7_1002_WHT_Security_Camera/SimpleHome_XCS7_1002_WHT_Security_Camera_attack_test.csv')
    Simple_02_test = (Simple_02_test - Simple_02_test.mean()) / (Simple_02_test.std())
    Simple_02_test = np.array(Simple_02_test)
    Simple_02_test[np.isnan(Simple_02_test)] = 0

    Simple_03_test = pd.read_csv(
        '/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/SimpleHome_XCS7_1003_WHT_Security_Camera/SimpleHome_XCS7_1003_WHT_Security_Camera_attack_test.csv')
    Simple_03_test = (Simple_03_test - Simple_03_test.mean()) / (Simple_03_test.std())
    Simple_03_test = np.array(Simple_03_test)
    Simple_03_test[np.isnan(Simple_03_test)] = 0

    test_data_num = len(Danmini_test) + len(Ecobee_test) + len(Ennio_test) + len(Philips_test) + len(Provision_73_test) + len(
        Provision_83_test) + len(Samsung_test) + len(Simple_02_test) + len(Simple_03_test)
    test_data_global = pd.read_csv('/Users/ultraz/PycharmProjects/FedDetect/data/UCI-MLR/attack_test.csv')
    test_data_global = (test_data_global - test_data_global.mean()) / (test_data_global.std())
    test_data_global = np.array(test_data_global)
    test_data_global[np.isnan(test_data_global)] = 0
    logging.info('Origin csv data is loaded, waiting for partition')
    return Danmini, Ecobee, Ennio, Philips, Provision_73, Provision_83, Samsung, Simple_02, Simple_03, train_data_num, train_data_global, \
           Danmini_test, Ecobee_test, Ennio_test, Philips_test, Provision_73_test, Provision_83_test, Samsung_test, Simple_02_test, Simple_03_test, \
           test_data_num, test_data_global


def local_dataloader(args):
    # Dict: records dataloaders for each devices
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()

    # get training and test set
    Danmini, Ecobee, Ennio, Philips, Provision_73, Provision_83, Samsung, Simple_02, Simple_03, train_data_num, train_data_global, \
    Danmini_test, Ecobee_test, Ennio_test, Philips_test, Provision_73_test, Provision_83_test, Samsung_test, Simple_02_test, Simple_03_test, \
    test_data_num, test_data_global = load_data(args)

    train_data_local_dict[0] = torch.utils.data.DataLoader(Danmini, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[1] = torch.utils.data.DataLoader(Ecobee, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[2] = torch.utils.data.DataLoader(Ennio, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[3] = torch.utils.data.DataLoader(Philips, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[4] = torch.utils.data.DataLoader(Provision_73, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[5] = torch.utils.data.DataLoader(Provision_83, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[6] = torch.utils.data.DataLoader(Samsung, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[7] = torch.utils.data.DataLoader(Simple_02, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    train_data_local_dict[8] = torch.utils.data.DataLoader(Simple_03, batch_size=args.batch_size,
                                                           shuffle=False,
                                                           num_workers=0)
    # for local train data
    for client_idx in range(9):
        train_data_local_num_dict[client_idx] = len(train_data_local_dict[client_idx])
        logging.info(
            "client_idx = %d, local_train_sample_number = %d" % (client_idx, train_data_local_num_dict[client_idx]))


    # for local test data
    test_data_local_dict[0] = torch.utils.data.DataLoader(Danmini_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[1] = torch.utils.data.DataLoader(Ecobee_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[2] = torch.utils.data.DataLoader(Ennio_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[3] = torch.utils.data.DataLoader(Philips_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[4] = torch.utils.data.DataLoader(Provision_73_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[5] = torch.utils.data.DataLoader(Provision_83_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[6] = torch.utils.data.DataLoader(Samsung_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[7] = torch.utils.data.DataLoader(Simple_02_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    test_data_local_dict[8] = torch.utils.data.DataLoader(Simple_03_test, batch_size=1,
                                                           shuffle=False,
                                                           num_workers=0)
    for client_idx in range(9):
        logging.info(
            "client_idx = %d, local_test_sample_number = %d" % (client_idx, len(test_data_local_dict[client_idx])))

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

    dataset = local_dataloader(args)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict] = dataset
    print(dataset)
