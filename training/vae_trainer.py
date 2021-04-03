import logging

import torch

from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class VAETrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        pass

    def test(self, test_data, device, args):

        return test_acc, test_total, test_loss

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False