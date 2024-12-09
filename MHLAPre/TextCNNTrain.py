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
import TextCNN as tc
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transfomer_data_ep_mhc1=torch.Tensor(np.load('ProcessData/transfomer_data_ep_mhc1.npy'))
transfomer_data_ep_mhc2=torch.Tensor(np.load('ProcessData/transfomer_data_ep_mhc2.npy'))
transfomer_data_ep_mhc3=torch.Tensor(np.load('ProcessData/transfomer_data_ep_mhc3.npy'))
transfomer_data_ep_mhc4=torch.Tensor(np.load('ProcessData/transfomer_data_ep_mhc4.npy'))
transfomer_data_ep_mhc5=torch.Tensor(np.load('ProcessData/transfomer_data_ep_mhc5.npy'))
transfomer_data_ep_mhc_train=np.concatenate((transfomer_data_ep_mhc1,transfomer_data_ep_mhc2,transfomer_data_ep_mhc3,transfomer_data_ep_mhc4,transfomer_data_ep_mhc5),axis=0)
transfomer_data_ep_mhc_train=torch.Tensor(transfomer_data_ep_mhc_train)
embedding_dim = 21  # Embedding dimension
kernel_sizes = [1, 2, 1]  # Different kernel sizes for convolutionspip
num_filters = 300  # Number of filters per kernel size
model = tc.TextCNN(embedding_dim, kernel_sizes, num_filters).to(device)



print(transfomer_data_ep_mhc_train.shape)

df = pd.read_csv('ProcessData/data_MHLAPre.csv')

label=list(df['Assay'])

X_train_tensor = (transfomer_data_ep_mhc_train).to(device)
y_train_tensor = torch.tensor(label, dtype=torch.float32).to(device)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor[:40000], y_train_tensor[:40000])

train_loader = DataLoader(train_dataset, batch_size=248, shuffle=True)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# model = Net().to(device)
fast_lr = 0.1  # 2
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# maml_qiao = learn2learn.algorithms.MAML(model, lr=fast_lr)
# 训练模型
num_epochs = 100  # 替换为您希望的训练轮数
loss_t = []
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        # 清除之前的梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)

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

# torch.save(maml_qiao.state_dict(), 'best_model.pth')


test_dataset = TensorDataset(X_train_tensor[40000:], y_train_tensor[40000:])

test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
i=0
with torch.no_grad():
    model.eval()
    correct = 0
    total = 0
    auc1=0
    roc_auc=0
    aupr=0
    fpr=0
    tpr=0
    prAUC=0
    item=0
    for inputs, labels in test_dataloader:
        outputs = model(inputs.data)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        try:
            i=i+1
            roc_auc += roc_auc_score(np.array(labels.cpu()),np.array(predicted.cpu()))
            fpr_result, tpr_result, _ = roc_curve(np.array(labels.cpu()),np.array(predicted.cpu()))
            auc1 += auc(fpr_result, tpr_result)
            # 计算PR曲线和AUPR
            precision, recall, _ = precision_recall_curve(np.array(labels.cpu()),np.array(predicted.cpu()))
            aupr += average_precision_score(np.array(labels.cpu()),np.array(predicted.cpu()))
        except ValueError:
            pass

    print(auc1/i)
    auc_sum=roc_auc/i
    aupr_sum=aupr/i
    prAUC_sum=prAUC/i
    accuracy = correct / total
    print('AUC:',auc_sum)
    print('AUPR:',aupr_sum)
    print('prAUC_sum:',prAUC_sum)
