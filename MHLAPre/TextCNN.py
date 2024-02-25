import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import HLA_encode as he
import learn2learn
import TransfomerEncoder as te

class TextCNN(nn.Module):
    def __init__(self, num_classes):
        super(TextCNN, self).__init__()

        # 一维卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=2)

        # # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2,stride=1)

        # 全连接层
        self.fc1 = nn.Linear(1020, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout层
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        print(x.shape)
        x = x.unsqueeze(1)  # 在第二个维度上增加一个维度，使其变为(batch_size, channels, sequence_length)
        x=x.reshape(x.size(0),1,21,-1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # 将特征展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


X_train_tensor = torch.Tensor(te.input_data)
y_train_tensor = torch.tensor(he.label[0:47826], dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_classes = 2

model = TextCNN(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
meta_lr = 0.00005  # 1
fast_lr = 0.005   # 2
maml_qiao = learn2learn.algorithms.MAML(model, lr=fast_lr)

# 训练模型
num_epochs = 250

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        # 清除之前的梯度
        optimizer.zero_grad()
        clone = maml_qiao.clone()
        # 前向传播
        outputs = clone(inputs.float())
        loss = criterion(outputs, labels.long())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 打印每个epoch的平均损失
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
