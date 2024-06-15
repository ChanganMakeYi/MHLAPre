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



transfomer_data_ep_mhc1=torch.Tensor(np.load('data/transfomer_data_ep_mhc1.npy'))
transfomer_data_ep_mhc2=torch.Tensor(np.load('data/data-mhlapre/transfomer_data_ep_mhc2.npy'))
transfomer_data_ep_mhc3=torch.Tensor(np.load('data/data-mhlapre/transfomer_data_ep_mhc3.npy'))
transfomer_data_ep_mhc4=torch.Tensor(np.load('data/data-mhlapre/transfomer_data_ep_mhc4.npy'))
transfomer_data_ep_mhc5=torch.Tensor(np.load('data/data-mhlapre/transfomer_data_ep_mhc5.npy'))
transfomer_data_ep_mhc=np.concatenate((transfomer_data_ep_mhc1,transfomer_data_ep_mhc2,transfomer_data_ep_mhc3,transfomer_data_ep_mhc4,transfomer_data_ep_mhc5),axis=0)
transfomer_data_ep_mhc=torch.Tensor(transfomer_data_ep_mhc)
embedding_dim = 21  # Embedding dimension
kernel_sizes = [1, 2, 3]  # Different kernel sizes for convolutionspip
num_filters = 300  # Number of filters per kernel size

model = TextCNN(embedding_dim, kernel_sizes, num_filters).to(device)


X_train_tensor = (transfomer_data_ep_mhc).to(device)
y_train_tensor = torch.tensor(label, dtype=torch.float32).to(device)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# model = Net().to(device)
fast_lr = 0.1  # 2
optimizer = optim.Adam(model.parameters(), lr=0.00005)
maml_qiao = learn2learn.algorithms.MAML(model, lr=fast_lr)
# 训练模型
num_epochs = 100  # 替换为您希望的训练轮数
loss_t = []
for epoch in tqdm(range(num_epochs)):
    maml_qiao.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        # 清除之前的梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = maml_qiao(inputs)

        loss = criterion(outputs, labels.long())
        #         print(loss.item())
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 打印每个epoch的平均损失
    average_loss = total_loss / len(train_loader)
    loss_t.append(average_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
