import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self, n_classes=2, num_features=2048, relu_in_last_fc=True):
        super().__init__()

        self.num_features = num_features
        self.relu_in_last_fc = relu_in_last_fc

        self.fc1 = nn.Linear(num_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, n_classes)
        self.bn4 = nn.BatchNorm1d(n_classes)  # Batch normalization after fc4

    def forward(self, x):
        x = x.view(-1, self.num_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Apply batch normalization after fc4 and before activation
        x = self.bn4(self.fc4(x))
        if self.relu_in_last_fc:
            x = F.relu(x)

        return x
