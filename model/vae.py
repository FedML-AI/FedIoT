import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, device):
        super(VAE, self).__init__()
        self.device = device
        # self.fc1 = nn.Linear(115, round(115 * 0.75))
        # self.fc21 = nn.Linear(round(115 * 0.75), round(115 * 0.50))
        # self.fc22 = nn.Linear(round(115 * 0.75), round(115 * 0.50))
        # self.fc3 = nn.Linear(round(115 * 0.50), round(115 * 0.33))
        # self.fc4 = nn.Linear(round(115 * 0.33), round(115 * 0.25))
        # self.fc5 = nn.Linear(round(115 * 0.25), round(115 * 0.33))
        # self.fc6 = nn.Linear(round(115 * 0.33), round(115 * 0.50))
        # self.fc7 = nn.Linear(round(115 * 0.50), round(115 * 0.75))
        # self.fc8 = nn.Linear(round(115 * 0.75), 115)
        # self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(115, round(115 * 0.75))
        self.fc2 = nn.Linear(round(115 * 0.75), round(115 * 0.50))
        self.fc3 = nn.Linear(round(115 * 0.50), round(115 * 0.33))
        self.fc41 = nn.Linear(round(115 * 0.33), round(115 * 0.25))
        self.fc42 = nn.Linear(round(115 * 0.33), round(115 * 0.25))
        self.fc5 = nn.Linear(round(115 * 0.25), round(115 * 0.33))
        self.fc6 = nn.Linear(round(115 * 0.33), round(115 * 0.50))
        self.fc7 = nn.Linear(round(115 * 0.50), round(115 * 0.75))
        self.fc8 = nn.Linear(round(115 * 0.75), 115)
        self.dropout = nn.Dropout(p=0.1)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2))
        h3 = self.dropout(h3)
        return self.fc41(h3), self.fc42(h3)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h5 = F.relu(self.fc5(z))
        h5 = self.dropout(h5)
        h6 = F.relu(self.fc6(h5))
        h6 = self.dropout(h6)
        h7 = F.relu(self.fc7(h6))
        h7 = self.dropout(h7)
        h8 = F.relu(self.fc8(h7))
        h8 = self.dropout(h8)
        return torch.sigmoid(h8)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
