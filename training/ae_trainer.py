import logging

import numpy as np
import torch
import torch.nn as nn
import wandb

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class AETrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info(device)
        model = self.model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        loss_func = torch.nn.MSELoss()
        for epoch in range(args.epochs):
            # mini- batch loop
            epoch_loss = 0.0
            for idx, inp in enumerate(train_data):
                inp = inp.to(device)
                optimizer.zero_grad()
                decode = model(inp)
                loss = loss_func(decode, inp)
                # epoch_loss += loss.item() / args.batch_size
                loss.backward()
                optimizer.step()

    def test(self, test_data, device, args):
        pass

    def test_local(self, client_index, b_global, threshold, test_data, device, args):
        model = self.model.to(device)
        self.model.eval()
        anmoaly = []
        thres_func = nn.MSELoss()
        for idx, inp in enumerate(test_data):
            inp = inp.to(device)
            decode = model(inp)
            diff = thres_func(decode, inp)
            if diff > threshold:
                anmoaly.append(idx)
        # batch size = 1 so we can use the batch number as the length
        #test_len = len(test_data)
        anmoaly_len = len(anmoaly)
        logging.info('(global = %s) client_index = %d, the local detected anmoaly number is %f' % (str(b_global), client_index, anmoaly_len))
        return anmoaly_len

    #how will adding a new parameter in the func influence the structure?
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        logging.info(device)
        threshold_dict = dict()
        diff_results_global = []
        thres_func = nn.MSELoss()
        for client_index in train_data_local_dict.keys():
            train_data = train_data_local_dict[client_index]
            diff_results_per_client = []
            self.model.eval()
            for idx, inp in enumerate(train_data):
                inp = inp.to(device)
                decode = self.model(inp)
                diff = thres_func(decode, inp)
                diff_results_per_client.append(diff)
                diff_results_global.append(diff)
            diff_results_per_client = torch.tensor(diff_results_per_client)
            threshold_dict[client_index] = (torch.mean(diff_results_per_client) + 0.8 * torch.std(diff_results_per_client)) / args.batch_size
        diff_results_global = torch.tensor(diff_results_global)
        threshold_global = (torch.mean(diff_results_global) + 0.8 * torch.std(diff_results_global)) / args.batch_size

        precision_array_global = []
        precision_array_local = []
        for client_index in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_index]

            # using global threshold for test
            precision_with_global_threshold_per_client = self.test_local(client_index, True, threshold_global, test_data, device, args)
            precision_array_global.append(precision_with_global_threshold_per_client)

            # using local threshold for test
            precision_with_local_threshold_per_client = self.test_local(client_index, False, threshold_dict[client_index], test_data, device, args)
            precision_array_local.append(precision_with_local_threshold_per_client)
        precision_mean_global = np.mean(precision_array_global)
        precision_mean_local = np.mean(precision_array_local)
        logging.info("precision_mean_global = %f" % precision_mean_global)
        logging.info("precision_mean_local = %f" % precision_mean_local)
        wandb.log({"Test/precision_mean_global": precision_mean_global})
        wandb.log({"Test/precision_mean_local": precision_mean_local})
        return True
