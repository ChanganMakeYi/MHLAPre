import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import Pretreatment as he
import learn2learn
#import TransfomerEncoder as te
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TextCNN(nn.Module):
    def __init__(self, embedding_dim, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        #
        # self.attention = nn.MultiheadAttention(50, 10)
        # self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size) for kernel_size in kernel_sizes])
        # # 池化层
        #
        # # 全连接层
        # self.fc1 = nn.Linear(900, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.fc2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.fc3 = nn.Linear(64, 2)
        # self.bn3 = nn.BatchNorm1d(2)
        # # Dropout层
        # self.dropout = nn.Dropout(0.2)

        self.attention = nn.MultiheadAttention(50, 10)
        self.fc = nn.Linear(50, 50)
        self.dense1 = nn.Linear(2100, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.dense2 = nn.Linear(1000, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dense3 = nn.Linear(100, 2)

        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        # # cov1d代码
        # x = x.transpose(1, 2)
        #
        # x, _ = self.attention(x, x, x)
        # pooled_outputs = []
        # x = x.reshape(-1, 21, 50)
        #
        # for conv in self.convs:
        #     conv_out = F.relu(conv(x))
        #     pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
        #     pooled_outputs.append(pooled)
        # x = torch.cat(pooled_outputs, dim=1)
        #
        # # 全连接层
        # x = F.relu(self.fc1(x))
        # #         x=self.bn1(x)
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.bn2(x)
        # x = F.relu(self.fc3(x))
        # x = self.bn3(x)
        # return F.softmax(x)
        #         print(1,ori_data.shape)
        x = x.transpose(1, 2)
        ori_data = x
        x, _ = self.attention(x, x, x)
        x = torch.cat((x, ori_data), dim=2)
        #         print(2,x.shape)
        #         x = torch.relu(self.fc(x))
        x = x.view(x.size(0), -1)
        #         print(3,x.shape)
        #         x = self.dropout(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout(x)
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)
        #         x = torch.softmax(x,dim=1)

        #         return torch.sigmoid(x)
        return F.softmax(x)





