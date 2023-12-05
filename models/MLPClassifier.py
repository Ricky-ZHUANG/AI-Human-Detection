import torch
from torch import nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, fc_in_dim=1500, fc_hid_dim=[512, 512], embed_dim=1, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.fc_hid_dim = fc_hid_dim
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()

        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                            #   nn.BatchNorm1d(output_size),
                            #   self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], embed_dim)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)

    def forward(self, x):
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = F.relu(fc(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x