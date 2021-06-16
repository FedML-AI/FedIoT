import argparse
import logging
import os
import random
import socket
import sys
import time
import requests

import numpy as np
import psutil
import setproctitle
import torch.nn
# import wandb


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from data_preprocessing.fl_dataloader import local_dataloader

#from FedML.fedml_api.distributed.utils.gpu_mapping import mapping_processes_to_gpu_device_from_yaml_file

from FedML.fedml_api.distributed.fedavg.FedAvgAPI import FedML_init, FedML_FedAvg_distributed

from FedML.fedml_api.distributed.fedavg.MyModelTrainer import MyModelTrainer
from FedML.fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer
from FedML.fedml_api.distributed.fedavg.FedAvgClientManager import FedAVGClientManager

from model.ae import AutoEncoder
from training.ae_trainer import AETrainer


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='vae', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='UCI-MIR', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../data/UCI-MLR',
                        help='data directory')

    # parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
    #                     help='how to partition the dataset on local workers')
    #
    # parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
    #                     help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=9, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=1, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=1,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--gpu_mapping_file', type=str, default=None,
                        help='the gpu utilization file for servers and clients. If there is no \
                        gpu_util_file, gpu will not be used.')

    parser.add_argument('--gpu_mapping_key', type=str, default="mapping_default",
                        help='the key in gpu utilization file')

    parser.add_argument('--grpc_ipconfig_path', type=str, default="grpc_ipconfig.csv",
                        help='config table containing ipv4 address of grpc server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--server_ip', type=str, default="http://192.168.3.86:5000",
                        help='IP address of the FedML server')

    parser.add_argument('--client_uuid', type=str, default="0",
                        help='number of workers in a distributed cluster')

    args = parser.parse_args()
    return args

def register(args, uuid):
    str_device_UUID = uuid
    URL = args.server_ip + "/api/register"

    # defining a params dict for the parameters to be sent to the API
    PARAMS = {'device_id': str_device_UUID}

    # sending get request and saving the response as response object
    r = requests.post(url=URL, params=PARAMS)
    result = r.json()
    client_ID = result['client_id']
    # executorId = result['executorId']
    # executorTopic = result['executorTopic']
    training_task_args = result['training_task_args']

    class Args:
        def __init__(self):
            self.dataset = training_task_args['dataset']
            self.data_dir = training_task_args['data_dir']
            self.partition_method = training_task_args['partition_method']
            self.partition_alpha = training_task_args['partition_alpha']
            self.model = training_task_args['model']
            self.client_num_per_round = training_task_args['client_num_per_round']
            self.comm_round = training_task_args['comm_round']
            self.epochs = training_task_args['epochs']
            self.lr = training_task_args['lr']
            self.wd = training_task_args['wd']
            self.batch_size = training_task_args['batch_size']
            self.frequency_of_the_test = training_task_args['frequency_of_the_test']
            self.is_mobile = training_task_args['is_mobile']

    args = Args()
    return client_ID, args

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device

def load_data(args, train_file_name, test_file_name):

    logging.info("load_train_data. dataset_name = %s" % train_file_name)
    logging.info("load_test_data. dataset_name = %s" % test_file_name)
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict\
        = local_dataloader(args, train_file_name, test_file_name, 1)
    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict]

    return dataset


def create_model(device):
    model = AutoEncoder()
    logging.info(model)
    return model


if __name__ == "__main__":
    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    main_args = add_args(parser)
    uuid = main_args.client_uuid
    client_ID, args = register(main_args, uuid)
    client_index = client_ID - 1

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    hostname = socket.gethostname()

    logging.info(args)
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
#     if process_id == 0:
#         wandb.init(project='fediot', entity='automl',
#                    name=str(args.model) + "r" + str(args.dataset) + "-lr" + str(args.lr),
#                    config=args)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_tensor_type(torch.DoubleTensor)

    # # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    # device = mapping_processes_to_gpu_device_from_yaml_file(process_id, worker_number, None,
    #                                                         args.gpu_mapping_key)
    device = init_training_device(client_ID - 1, args.client_num_per_round - 1, 4)

    # load data

    dataset = load_data(main_args, 'federated_learning_data/train_unified_10.csv', 'new_centralized_set/global_testset_test.csv')
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(device)

    # create my own trainer
    model_trainer = AETrainer(model)
    # client_index = 1
    # start training
    trainer = FedAVGTrainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                            train_data_num, device,
                            args, model_trainer)

    size = main_args.client_num_per_round + 1
    client_manager = FedAVGClientManager(args, trainer, rank=client_ID, size=size, backend="MQTT")
    client_manager.run()
    client_manager.start_training()

    time.sleep(100000)


