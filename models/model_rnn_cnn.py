# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, bi_value, drop, kernel_size, kernel_size2, 
                 stride, stride2, out_channels_first, out_channels_second, linear_first):
        super(RNN_CNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bi_value)
        if bi_value == True:
          self.conv1 = nn.Conv1d(in_channels=hidden_dim*2, out_channels=out_channels_first, kernel_size=kernel_size, stride=stride, padding=1)
        else:
          self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=out_channels_first, kernel_size=kernel_size, stride=stride, padding=1)

        self.conv2 = nn.Conv1d(in_channels=out_channels_first, out_channels=out_channels_second, kernel_size=kernel_size2, stride=stride2, padding=1)
        self.lin1 = nn.Linear(out_channels_second, linear_first)
        self.lin2 = nn.Linear(linear_first, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=drop)

    def forward(self, X):
        input_ = X.unsqueeze(dim=1)
        lstm_out, (h, c) = self.lstm(input_)

        X = self.dropout(lstm_out)

        X = X.transpose(1, 2)
        x_cnn1 = self.conv1(X)
        x_cnn2 = self.conv2(x_cnn1)
        X = torch.squeeze(x_cnn2)

        X = self.dropout(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.sigmoid(X)

        return X