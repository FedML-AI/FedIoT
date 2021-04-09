import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Linear(115, round(115*0.75)),
            nn.Tanh(),
            nn.Dropout(p = 0.1),
            nn.Linear(round(115*0.75), round(115*0.50)),
            nn.Tanh(),
            nn.Dropout(p = 0.1),
            nn.Linear(round(115*0.50), round(115*0.33)),
            nn.Tanh(),
            nn.Dropout(p = 0.1),
            nn.Linear(round(115*0.33), round(115*0.25)),
            nn.Tanh()
        )
        self.dec = nn.Sequential(
            nn.Linear(round(115*0.25), round(115*0.33)),
            nn.Tanh(),
            nn.Dropout(p = 0.1),
            nn.Linear(round(115*0.33), round(115*0.50)),
            nn.Tanh(),
            nn.Dropout(p = 0.1),
            nn.Linear(round(115*0.50), round(115*0.75)),
            nn.Tanh(),
            nn.Dropout(p = 0.1),
            nn.Linear(round(115*0.75), 115),
            nn.Tanh()
        )
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
