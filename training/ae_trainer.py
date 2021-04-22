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
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_func = torch.nn.MSELoss()
        # model training
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

    def test_local(self, client_index, opt_idx_threshold, threshold, test_data, device, args):
        model = self.model.to(device)
        self.model.eval()

        true_negative = []
        false_positive = []
        true_positive = []
        false_negative = []
        thres_func = nn.MSELoss()

        for idx, inp in enumerate(test_data):
            inp = inp.to(device)
            decode = model(inp)
            diff = thres_func(decode, inp)
            mse = diff.item()
            if idx < opt_idx_threshold:
                if mse > threshold:
                    false_positive.append(idx)
                else:
                    true_negative.append(idx)
            else:
                if mse > threshold:
                    true_positive.append(idx)
                else:
                    false_negative.append(idx)

        accuracy = (len(true_positive) + len(true_negative)) \
                   / (len(true_positive) + len(true_negative) + len(false_positive) + len(false_negative))
        precision = len(true_positive) / (len(true_positive) + len(false_positive))
        false_positive_rate = len(false_positive) / (len(false_positive) + len(true_negative))

        logging.info('client_index = %d, The True negative number is %f' % (client_index, len(true_negative)))
        logging.info('client_index = %d, The False positive number is %f' % (client_index, len(false_positive)))
        logging.info('client_index = %d, The True positive number is %f' % (client_index, len(true_positive)))
        logging.info('client_index = %d, The False negative number is %f' % (client_index, len(false_negative)))
        logging.info('client_index = %d, The accuracy is %f' % (client_index, accuracy))
        logging.info('client_index = %d, The precision is %f' % (client_index, precision))
        logging.info('client_index = %d, The false positive rate is %f' % (client_index, false_positive_rate))

        return accuracy, precision, false_positive_rate

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None):
        logging.info(device)
        #mse_results_global = []
        threshold_dict = {}
        thres_func = nn.MSELoss()
        opt_threshold = [round(49548.0 * 0.67 / args.batch_size), round(13113.0 * 0.67 / args.batch_size),
                         round(39100.0 * 0.67 / args.batch_size), round(175240.0 * 0.67 / args.batch_size),
                         round(62154.0 * 0.67 / args.batch_size), round(98514.0 * 0.67 / args.batch_size),
                         round(52150.0 * 0.67 / args.batch_size), round(46585.0 * 0.67 / args.batch_size),
                         round(19528.0 * 0.67 / args.batch_size)]
        for client_index in train_data_local_dict.keys():
            opt_data = train_data_local_dict[client_index]
            mse_results_per_client = []
            self.model.eval()
            for idx, inp in enumerate(opt_data):
                if idx >= opt_threshold[client_index]:
                    inp = inp.to(device)
                    decode = self.model(inp)
                    diff = thres_func(decode, inp)
                    mse = diff.item()
                    mse_results_per_client.append(mse)
                    #mse_results_global.append(mse)
            mse_results_per_client = torch.tensor(mse_results_per_client)
            threshold_dict[client_index] = (torch.mean(mse_results_per_client) + 1 * torch.std(mse_results_per_client)) / np.sqrt(
            args.batch_size)
        # mse_results_global = torch.tensor(mse_results_global)
        # threshold_global = (torch.mean(mse_results_global) + 1 * torch.std(mse_results_global)) / np.sqrt(
        #     args.batch_size)

        accuracy_array_global = []
        precision_array_global = []
        fpr_array_global = []

        for client_index in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_index]

            # using global threshold for test
            [accuracy_client, precision_client, fpr_client] = self.test_local(client_index,
                                                                              (opt_threshold[client_index] / 2), threshold_dict[client_index], test_data, device, args)
            accuracy_array_global.append(accuracy_client)
            precision_array_global.append(precision_client)
            fpr_array_global.append(fpr_client)

            # # using local threshold for test
            # precision_with_local_threshold_per_client = self.test_local(client_index, False, threshold_dict[client_index], test_data, device, args)
            # precision_array_local.append(precision_with_local_threshold_per_client)

        accuracy_mean_global = np.mean(accuracy_array_global)
        precision_mean_global = np.mean(precision_array_global)
        fpr_mean_global = np.mean(fpr_array_global)

        # precision_mean_local = np.mean(precision_array_local)
        logging.info("accuracy_mean_global = %f" % accuracy_mean_global)
        logging.info("precision_mean_global = %f" % precision_mean_global)
        logging.info("fpr_mean_global = %f" % fpr_mean_global)

        wandb.log({"Test/accuracy_mean_global": accuracy_mean_global})
        wandb.log({"Test/precision_mean_global": precision_mean_global})
        wandb.log({"Test/fpr_mean_global": fpr_mean_global})

        return True
