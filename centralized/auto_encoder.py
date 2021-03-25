import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(115, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 115),
            nn.ReLU()
        )

    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
