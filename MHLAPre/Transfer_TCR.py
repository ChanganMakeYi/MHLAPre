import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import TextCNN as TextCNNmodel
import Concatenate_CDR3_EPI
model = TextCNNmodel.model
# 冻结BERT模型中的所有层
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层的权重，使其能够适应新的任务
num_ftrs = model.dense3.in_features
model.dense3 = nn.Linear(num_ftrs, 2)

HLA_all=pd.read_csv('data/TCR-HLA-epotite4.csv')

Label=pd.Series(list(HLA_all['Label']))

X_train_tensor = torch.Tensor(Concatenate_CDR3_EPI.EP_TCR_MHC)
y_train_tensor = torch.tensor(Label, dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
#
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels.long())

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 打印每个epoch的平均损失
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')
