import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, drop, kernel_size, kernel_size2, stride, stride2, out_channels_first, out_channels_second, linear_first):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels_first, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(in_channels=out_channels_first, out_channels=out_channels_second, kernel_size=kernel_size2, stride=stride2, padding=1)
        self.lin1 = nn.Linear(out_channels_second, linear_first)
        self.lin2 = nn.Linear(linear_first, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(p=drop)

    def forward(self, X):
        X = torch.unsqueeze(X, 1)
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