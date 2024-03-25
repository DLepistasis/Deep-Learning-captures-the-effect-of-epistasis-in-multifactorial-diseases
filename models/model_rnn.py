import torch
import torch.nn as nn


class Model(nn.Module)::
    def __init__(self, input_dim, hidden_dim, target_size, dropout, bi_value):
        super(LSTMClassification, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bi_value)
        self.fc = nn.Linear(hidden_dim, target_size)
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_):
        input_ = input_.unsqueeze(dim=1)
        lstm_out, (h, c) = self.lstm(input_)
        lstm_out = h[-1]
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        scores = self.sigmoid(logits)
        return scores