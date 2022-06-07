from math import sqrt

import torch
import torch.nn as nn


def cal_sharpe_ratio_in_torch(weight, future_return, trans_rate):
    n = weight.shape[0]
    prev_weight = torch.cat((torch.zeros(1, weight.shape[1]), weight[:-1, :]), dim=0)

    cost = trans_rate * torch.abs(weight - prev_weight)
    e_r = torch.mean(torch.sum(future_return * weight - cost, dim=1))
    std = sqrt((n - 1) / n) * torch.std(torch.sum(future_return * weight - cost, dim=1))

    sharpe = e_r / std
    return sharpe


class DeepLOB(nn.Module):
    def __init__(self, y_len, trans_rate):
        super().__init__()
        self.y_len = y_len

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3), stride=(1, 3)),
            nn.LeakyReLU(negative_slope=0.01),
            #             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 3)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

        self.trans_rate = trans_rate

    def forward(self, data):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # x, future_return = input

        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, data.size(0), 64).to(device)
        c0 = torch.zeros(1, data.size(0), 64).to(device)
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)

        x_inp1 = self.inp1(data)
        x_inp2 = self.inp2(data)
        x_inp3 = self.inp3(data)

        data = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        #         x = torch.transpose(x, 1, 2)
        data = data.permute(0, 2, 1, 3)
        data = torch.reshape(data, (-1, data.shape[1], data.shape[2]))

        data, _ = self.lstm(data, (h0, c0))
        data = data[:, -1, :]
        w = self.fc1(data)

        w = torch.softmax(w, dim=1)
        return w


