import logging
import os
import sys

import argparse
import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from FedML.fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from FedML.fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager
from FedML.fedml_api.distributed.fedavg.MyModelTrainer import MyModelTrainer

from data_preprocessing.fl_dataloader import local_dataloader
from model.ae import AutoEncoder
from training.ae_trainer import AETrainer

from FedML.fedml_api.model.cv.mobilenet import mobilenet
from FedML.fedml_api.model.cv.resnet import resnet56
from FedML.fedml_api.model.linear.lr import LogisticRegression
from FedML.fedml_api.model.nlp.rnn import RNN_OriginalFedAvg

from FedML.fedml_core.distributed.communication.observer import Observer

from flask import Flask, request, jsonify, send_from_directory, abort


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lr', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='mnist', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../data/UCI-MLR',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--client_num_in_total', type=int, default=9, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=1, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')

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

    parser.add_argument('--ci', type=int, default=0,
                        help='continuous integration')

    parser.add_argument('--is_preprocessed', type=bool, default=False, help='True if data has been preprocessed')

    parser.add_argument('--grpc_ipconfig_path', type=str, default="../executor/grpc_ipconfig.csv",
                        help='config table containing ipv4 address of grpc server')

    args = parser.parse_args()
    return args


# HTTP server
app = Flask(__name__)
app.config['MOBILE_PREPROCESSED_DATASETS'] = './preprocessed_dataset/'

# parse python script input parameters
parser = argparse.ArgumentParser()
args = add_args(parser)

device_id_to_client_id_dict = dict()


@app.route('/', methods=['GET'])
def index():
    return 'backend service for Fed_mobile'


@app.route('/get-preprocessed-data/<dataset_name>', methods = ['GET'])
def get_preprocessed_data(dataset_name):
    directory = app.config['MOBILE_PREPROCESSED_DATASETS'] + args.dataset.upper() + '_mobile_zip/'
    try:
        return send_from_directory(
            directory,
            filename=dataset_name + '.zip',
            as_attachment=True)

    except FileNotFoundError:
        abort(404)


@app.route('/api/register', methods=['POST'])
def register_device():
    global device_id_to_client_id_dict
    # __log.info("register_device()")
    device_id = request.args['device_id']
    registered_client_num = len(device_id_to_client_id_dict)
    if device_id in device_id_to_client_id_dict:
        client_id = device_id_to_client_id_dict[device_id]
    else:
        client_id = registered_client_num + 1
        device_id_to_client_id_dict[device_id] = client_id

    training_task_args = {"dataset": args.dataset,
                          "data_dir": args.data_dir,
                          "partition_method": args.partition_method,
                          "partition_alpha": args.partition_alpha,
                          "model": args.model,
                          "client_num_per_round": args.client_num_per_round,
                          "comm_round": args.comm_round,
                          "epochs": args.epochs,
                          "lr": args.lr,
                          "wd": args.wd,
                          "batch_size": args.batch_size,
                          "frequency_of_the_test": args.frequency_of_the_test,
                          "is_mobile": args.is_mobile,
                          'dataset_url': '{}/get-preprocessed-data/{}'.format(
                              request.url_root,
                              client_id-1
                          ),
                          'is_preprocessed': args.is_preprocessed,
                          'grpc_ipconfig_path': args.grpc_ipconfig_path}

    return jsonify({"errno": 0,
                    "executorId": "executorId",
                    "executorTopic": "executorTopic",
                    "client_id": client_id,
                    "training_task_args": training_task_args})

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



if __name__ == '__main__':
    # MQTT client connection
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    logging.info(args)

    wandb.init(
        # project="federated_nas",
        project="fedml",
        name="mobile(mqtt)" + str(args.partition_method) + "r" + str(args.comm_round) + "-e" + str(
            args.epochs) + "-lr" + str(
            args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    # GPU 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    dataset = load_data(args, "fedrated_learning_data/train_unified.csv",
                        "new_centralized_set/global_testset_test.csv")
    [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(device)
    model_trainer = AETrainer(model)

    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                  args.client_num_per_round, device, args, model_trainer)
    size = args.client_num_per_round + 1
    server_manager = FedAVGServerManager(args,
                                         aggregator,
                                         rank=0,
                                         size=size,
                                         backend="MQTT",
                                         is_preprocessed=args.is_preprocessed)
    server_manager.run()

    # if run in debug mode, process will be single threaded by default
    app.run(host="192.168.3.86", port=5000)