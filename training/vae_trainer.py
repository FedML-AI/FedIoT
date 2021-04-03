import logging

import numpy as np
import torch
import wandb

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class VAETrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def loss_function_vae(self, recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """
        reconstruction_function = torch.nn.BCELoss(size_average=False)  # mse loss
        BCE = reconstruction_function(recon_x, x)
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        # KL divergence
        return BCE + KLD

    def train(self, train_data, device, args):
        model = self.model
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            # mini- batch loop
            epoch_loss = 0.0
            for idx, inp in enumerate(train_data):
                inp = inp.to(device)
                optimizer.zero_grad()
                decode, mu, logvar = model(inp)
                loss = self.loss_function_vae(decode, inp, mu, logvar)
                logging.info(loss)
                epoch_loss += loss.item() / args.batch_size
                loss.backward()
                optimizer.step()
                logging.info("epoch = %d, epoch_loss = %f" % (epoch, epoch_loss))
        logging.info("batch size = %d" % args.batch_size)

    def test(self, test_data, device, args):
        pass

    def test_local(self, threshold, test_data, device, args):
        model = self.model
        anmoaly = []
        for idx, inp in enumerate(test_data):
            inp = inp.to(device)
            decode, mu, logvar = model(inp)
            diff = torch.sum(abs(inp - decode))
            if diff > threshold:
                anmoaly.append(idx)
        # batch size = 1 so we can use the batch number as the length
        test_len = len(test_data)
        precision = (len(anmoaly) / test_len)
        print('The local precision is ', precision)
        return precision

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:

        threshold_dict = dict()
        diff_results_global = []
        for client_index in train_data_local_dict.key():
            train_data = train_data_local_dict[client_index]
            diff_results_per_client = []
            self.model.eval()
            for idx, inp in enumerate(train_data):
                inp = inp.to(device)
                decode, mu, logvar = self.model(inp)
                diff = torch.sum(abs(decode - inp))
                diff_results_per_client.append(diff)
                diff_results_global.append(diff)
            diff_results_per_client = torch.tensor(diff_results_per_client)
            threshold_dict[client_index] = (torch.mean(diff_results_per_client) + 0.8 * torch.std(diff_results_per_client)) / args.batch_size
        diff_results_global = torch.tensor(diff_results_global)
        threshold_global = (torch.mean(diff_results_global) + 0.8 * torch.std(diff_results_global)) / args.batch_size

        precision_array_global = []
        precision_array_local = []
        for client_index in test_data_local_dict.key():
            test_data = test_data_local_dict[client_index]

            # using global threshold for test
            precision_with_global_threshold_per_client = self.test_local(threshold_global, test_data, device, args)
            precision_array_global.append(precision_with_global_threshold_per_client)

            # using local threshold for test
            precision_with_local_threshold_per_client = self.test_local(threshold_dict[client_index], test_data, device, args)
            precision_array_local.append(precision_with_local_threshold_per_client)
        precision_mean_global = np.mean(precision_array_global)
        precision_mean_local = np.mean(precision_array_local)
        logging.info("precision_mean_global = %f" % precision_mean_global)
        logging.info("precision_mean_local = %f" % precision_mean_local)
        wandb.log({"Test/precision_mean_global": precision_mean_global})
        wandb.log({"Test/precision_mean_local": precision_mean_local})
        return True
