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
import TransfomerEncoder as te


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()

        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, kernel_size) for kernel_size in kernel_sizes])
        # 池化层

        # 全连接层
        self.fc1 = nn.Linear(900, 2048)
        #         self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1048)
        self.bn2 = nn.BatchNorm1d(1048)
        self.fc3 = nn.Linear(1048, 2)
        self.bn3 = nn.BatchNorm1d(2)
        # Dropout层
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):


        # cov1d代码
        x = x.transpose(1, 2)

        x, _ = self.attention(x, x, x)
        pooled_outputs = []
        x = x.reshape(-1, 21, 50)

        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            pooled_outputs.append(pooled)

        x = torch.cat(pooled_outputs, dim=1)

        x = (x.reshape(128, -1))[:][600:]

        # 全连接层
        x = F.relu(self.fc1(x))
        #         x=self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        return F.softmax(x)


X_train_tensor = torch.Tensor(te.input_data)
y_train_tensor = torch.tensor(he.label[0:47826], dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

num_classes = 2

model = TextCNN(num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)
fast_lr = 0.1
maml_qiao = learn2learn.algorithms.MAML(model, lr=fast_lr)

# 训练模型
num_epochs = 350

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
