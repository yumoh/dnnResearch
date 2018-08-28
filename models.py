import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderType1(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            nn.Linear(dim_in, dim_in * 2),
            nn.PReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(dim_in * 2, dim_in * 4),
            nn.PReLU()
        )

        self.layer3 = nn.Linear(dim_in * 4, dim_in)

    def forward(self, *input):
        x = input[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def encode(self, x):
        x1 = self.layer1(x)
        yield x1
        x2 = self.layer2(x1)
        yield x2


class EncoderType2(nn.Module):
    def __init__(self, dim_in):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            nn.Linear(dim_in, dim_in * 4),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(dim_in * 4, int(dim_in * 0.8)),
            nn.Tanh(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(int(dim_in * 0.8), int(dim_in * 1.2)),
            nn.PReLU()
        )
        self.layer4 = nn.Linear(int(dim_in * 0.4), dim_in)

    def forward(self, *input):
        x = input[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def encode(self, x):
        x1 = self.layer1(x)
        yield x1
        x2 = self.layer2(x1)
        yield x2
        x3 = self.layer3(x2)
        yield x3


class AutoEncodeDecode:
    def __init__(self, x):
        self.dim_in = x.shape[-1]
        self.x = torch.tensor(x, device=device, dtype=torch.float32)
        self.x_scale = x / x.max()

        self.net1 = EncoderType1(self.dim_in).to(torch.float32).to(device=device)
        self.net2 = EncoderType2(self.dim_in).to(torch.float32).to(device=device)

    def encode(self):
        self.fit_model(self.net1)
        self.fit_model(self.net2)

    def fit_model(self, net):
        net.train()

        dim_in = self.dim_in
        x = self.x
        x_scale = self.x_scale

        loss_table = [1]
        lr = 1.0
        optimizer = optim.Adadelta(net.parameters(), lr=lr)

        bar = tqdm(range(1000))
        for loop in bar:

            optimizer.zero_grad()
            out = net(x)
            scale_out = out / out.max()

            loss = F.l1_loss(scale_out, x_scale)
            loss.backward()
            optimizer.step()

            loss_table.append(loss.item())
            if loop % 80 == 0 and loop > 0:
                lr = loss.item() * 10
                optimizer = optim.Adadelta(net.parameters(), lr=lr)

            if loss.item() < 0.01:
                break

            bar.set_postfix(l=f'{loss.item():0.4f}')
        bar.close()

    def forward(self, *input):
        yield from self.net1.encode(input[0])
        yield from self.net2.encode(input[1])


class DNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
