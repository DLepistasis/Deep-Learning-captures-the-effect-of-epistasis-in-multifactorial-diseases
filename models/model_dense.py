import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, bn_momentum=None, first_dropout=None, other_dropouts=None,
                 lin1_output=None, lin2_output=None, lin3_output=None):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(input_size, lin1_output)
        self.bn1 = nn.BatchNorm1d(lin1_output, momentum=bn_momentum)
        self.lin2 = nn.Linear(lin1_output, lin2_output)
        self.bn2 = nn.BatchNorm1d(lin2_output, momentum=bn_momentum)

        self.lin3 = nn.Linear(lin2_output, lin3_output)
        self.bn3 = nn.BatchNorm1d(lin3_output, momentum=bn_momentum)

        self.lin4 = nn.Linear(lin3_output, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.dropout_first = nn.Dropout(p=first_dropout)
        self.dropout_other = nn.Dropout(p=other_dropouts)

    def forward(self, X):
        X = self.lin1(X)
        X = self.relu(X)
        X = self.bn1(X)
        X = self.dropout_first(X)

        X = self.lin2(X)
        X = self.relu(X)
        X = self.bn2(X)
        X = self.dropout_other(X)

        X = self.lin3(X)
        X = self.relu(X)
        X = self.bn3(X)

        X = self.dropout_other(X)

        X = self.lin4(X)
        X = self.sigmoid(X)

        return X
