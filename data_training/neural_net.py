import torch.nn.functional as F
from torch import nn


class ClassifierModule(nn.Module):

    def __init__(
        self,
        num_features=5,
        num_units=1024,
        n_classes=2,
        nonlin=F.relu,
        dropout=0.1,
        depth=2,
        batchnorm=True,
    ):
        super(ClassifierModule, self).__init__()
        self.num_features = num_features
        self.num_units = num_units
        self.n_classes = n_classes
        self.nonlin = nonlin
        self.batchnorm = batchnorm
        self.depth = depth

        self.dense0 = nn.Linear(self.num_features, self.num_units)
        self.nonlin = self.nonlin
        self.dropout = nn.Dropout(dropout)

        layers = []
        for i in range(1, self.depth):
            layers.append(nn.Linear(self.num_units, self.num_units))
        self.dense1 = nn.Sequential(*layers)

        self.output = nn.Linear(self.num_units, self.n_classes)
        self.bn = nn.BatchNorm1d(self.n_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)

        if self.batchnorm:
            X = self.bn(X)

        X = F.softmax(X, dim=-1)
        return X
