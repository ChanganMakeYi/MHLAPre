from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import Pretreatment as pr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.fc1 = nn.Linear(925, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x