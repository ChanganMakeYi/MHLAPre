import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
print(1)
import TextCNN as tc
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
import torch
from collections import OrderedDict
from tqdm import tqdm
import learn2learn
print(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embedding_dim = 21  # Embedding dimension
kernel_sizes = [1, 2, 1]  # Different kernel sizes for convolutionspip
num_filters = 300  # Number of filters per kernel size
model = tc.TextCNN(embedding_dim, kernel_sizes, num_filters).to(device)


#
# # 加载保存的 state_dict
# state_dict = torch.load('best_model.pth')
#
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k.replace("module.", "")  # 去掉 'module.' 前缀
#     new_state_dict[name] = v
#
# # 加载修改后的 state_dict
# model.load_state_dict(new_state_dict)
# for param in model.parameters():
#     param.requires_grad = False
#
# num_ftrs = model.dense3.in_features
# model.dense3 = nn.Linear(num_ftrs, 2)


model=model.to(device)
print(model)
# 冻结BERT模型中的所有层


HLA_all=pd.read_csv('data/train_paired_tcr_pmhc_data.csv')

Label=pd.Series(list(HLA_all['Target']))
EP_TCR_MHC=np.load("ProcessData/Transfer_data/hla_epit_cdr3.npy")
X_train_tensor = torch.Tensor(EP_TCR_MHC)
y_train_tensor = torch.tensor(Label, dtype=torch.float32)

X_train_tensor=X_train_tensor.to(device)
y_train_tensor=y_train_tensor.to(device)
# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor[:60000], y_train_tensor[:60000])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

print(X_train_tensor.device)
print(next(model.parameters()).device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
loss_fn = torch.nn.CrossEntropyLoss()
#
fast_lr=0.01
print(1111)
num_epochs = 30
maml_qiao = learn2learn.algorithms.MAML(model, lr=fast_lr)
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # 确保模型在正确的设备上
        maml_qiao = maml_qiao.to(device)

        # 确保输入数据也在相同的设备上
        inputs = inputs.to(device)

        # 进行前向传播
        outputs = maml_qiao(inputs)
        loss = loss_fn(outputs, labels.long())

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 打印每个epoch的平均损失
    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

test_dataset = TensorDataset(X_train_tensor[50000:], y_train_tensor[50000:])
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
        outputs = maml_qiao(inputs.data)
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
