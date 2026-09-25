from tqdm import tqdm
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import learn2learn
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
import Pretreatment as pr
import DataPre as dp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#训练模块代码
f = open("encode/embedding_32.txt", "r")
lines = f.readlines()  # 读取全部内容
for line in lines[1:21]:
    li = line.split(',')
    myarray = np.array(li[1:])
    myarray = myarray.astype(float)
    # Bugfix (not a reconstruction, see RESTORATION_NOTES.md): `import DataPre
    # as dp` above already runs this exact concatenation once on the same
    # shared pr.aa_dict_atchley dict; without this guard it runs a second
    # time here and corrupts every vector from 37 to 69 dims, which then
    # breaks every fixed-size reshape/Linear layer downstream.
    if len(pr.aa_dict_atchley[li[0]]) == 5:
        pr.aa_dict_atchley[li[0]] = np.concatenate((pr.aa_dict_atchley[li[0]], myarray), axis=0)
torch.set_default_dtype(torch.float64)
# antigen_array 抗原数据
model=dp.CNNModule().to(device)
model2=dp.CNNModule2().to(device)
TCR_list, antigen_list = pr.preprocess("data/train.csv")
antigen_array_blosum = pr.antigenMap(antigen_list, 15, 'BLOSUM50')  # 抗原序列最长为11
TCR_array_blosum = pr.antigenMap(TCR_list, 25, 'BLOSUM50')
antigen_array_blosum = torch.Tensor(antigen_array_blosum)
TCR_array_blosum = torch.Tensor(TCR_array_blosum)

TCR_antigen_result_blosum_ori = torch.cat((antigen_array_blosum, TCR_array_blosum), dim=1)  # TCR序列最长为24
TCR_antigen_result_blosum = TCR_antigen_result_blosum_ori.reshape(len(antigen_list), 21, -1)
print(TCR_antigen_result_blosum.shape)
TCR_antigen_result_blosum = TCR_antigen_result_blosum.to(device)
antigen_array = pr.aamapping_TCR(antigen_list, pr.aa_dict_atchley, 15)
print(antigen_array.shape)

antigen_array = torch.Tensor(antigen_array)
antigen_array = antigen_array.to(device)

antigen_array = antigen_array.reshape(len(antigen_list), 37, -1)
antigen_result = torch.empty(1, 16, 15)


print("2%%%%%%%%%%%%")
TCR_array = pr.aamapping_TCR(TCR_list, pr.aa_dict_atchley, 25)
TCR_array = torch.Tensor(TCR_array)
TCR_array = TCR_array.to(device)
TCR_array = TCR_array.reshape(len(TCR_list), 37, -1)

print("3%%%%%%%%%%%%")

TCR_result = torch.empty(1, 16, 15)
# for i in tqdm(range(0, len(TCR_array))):
#     if i == 0:
#         TCR_array1 = TCR_array[i].reshape(1, -1)
#         print(TCR_array1.shape)
#         output = model(TCR_array1)
#         TCR_result = output
#     else:
#         TCR_array1 = TCR_array[i].reshape(1, -1)
#         output = model(TCR_array1)
#         TCR_result = torch.cat((TCR_result, output), dim=0)


TCR_antigen_result=torch.cat((antigen_array,TCR_array),dim=2)
print(TCR_antigen_result.shape)
print(TCR_antigen_result[0])
# TCR_antigen_result=TCR_antigen_result.reshape(23232,-1)
print(TCR_antigen_result.shape)

TCR_antigen_result_sum=torch.cat((TCR_antigen_result,TCR_antigen_result_blosum),dim=1)
print("   111111111",TCR_antigen_result_sum.shape)

# Reconstructed (not recovered from git history, see RESTORATION_NOTES.md).
# `Net` is the pre-refactor classifier for the pHLA-TCR transfer stage, later
# replaced by reusing the TextCNN class directly in Transfer_TCR.py. It
# mirrors that class's attention + tri-layer-FC + softmax pattern, sized for
# the (batch, 58, 40) TCR_antigen_result_sum input built above (58 = 37 + 21
# rows, 40 = 15 + 25 antigen/TCR columns).
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.attention = nn.MultiheadAttention(40, 10)
        self.dense1 = nn.Linear(58 * 80, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.dense2 = nn.Linear(1000, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.dense3 = nn.Linear(100, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        ori_data = x
        x, _ = self.attention(x, x, x)
        x = torch.cat((x, ori_data), dim=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.dense1(x))
        x = self.dropout(x)
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)
        return torch.softmax(x, dim=1)


# 实例化模型
model_pre = Net().to(device)
meta_lr = 0.00005  # 1
fast_lr = 0.005  # 2
maml_qiao = learn2learn.algorithms.MAML(model_pre, lr=fast_lr)

label = pr.get_label("data/train.csv")
label = torch.Tensor(label)
# label=torch.unsqueeze(label,1)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(maml_qiao.parameters(), lr=meta_lr)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
label = label.to(device)
# 加载数据

TCR_antigen_result = dp.TCR_antigen_result.to(device)
dataset = TensorDataset(dp.TCR_antigen_result_sum, label)
dataloader = DataLoader(dataset, batch_size=64)

loss_sum = np.zeros(200)
# 训练模型
for epoch in tqdm(range(200)):
    for batch_x, batch_y in dataloader:
        clone = maml_qiao.clone()
        output = clone(batch_x.data)
        # output = model_pre()
        loss = criterion(output, batch_y.long())
        clone.adapt(loss, allow_unused=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    loss_sum[epoch] = loss.item()


def plot_loss(loss_values):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'b', label='Training loss')
    plt.title('Loss Function Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
plot_loss(loss_sum)

print(loss_sum)

#测试模块代码
TCR_list2,antigen_list2=pr.preprocess("data/test.csv")

antigen_array_blosum2=pr.antigenMap(antigen_list2,15,'BLOSUM50')
TCR_array_blosum2=pr.antigenMap(TCR_list2,25,'BLOSUM50')
antigen_array_blosum2=torch.Tensor(antigen_array_blosum2)
print(antigen_array_blosum2.shape)
TCR_array_blosum2=torch.Tensor(TCR_array_blosum2)
print(TCR_array_blosum2.shape)

TCR_antigen_result_blosum_ori2=torch.cat((antigen_array_blosum2,TCR_array_blosum2),dim=1)
print(TCR_antigen_result_blosum_ori2.shape)
TCR_antigen_result_blosum2=TCR_antigen_result_blosum_ori2.reshape(len(antigen_list2),21,-1)
print(TCR_antigen_result_blosum2.shape)
TCR_antigen_result_blosum2=TCR_antigen_result_blosum2.to(device)

antigen_array2=pr.aamapping_TCR(antigen_list2,pr.aa_dict_atchley,15)

antigen_array2=torch.DoubleTensor(antigen_array2)
antigen_array2=antigen_array2.to(device)
antigen_array2 = antigen_array2.reshape(len(antigen_list2), 37,-1)

print("2%%%%%%%%%%%%")
TCR_array2 = pr.aamapping_TCR(TCR_list2, pr.aa_dict_atchley, 25)
TCR_array2 = torch.DoubleTensor(TCR_array2)
TCR_array2=TCR_array2.to(device)
TCR_array2 = TCR_array2.reshape(len(TCR_list2),37, -1)
print(TCR_array2.shape)

print("3%%%%%%%%%%%%")

TCR_antigen_result2=torch.cat((antigen_array2,TCR_array2),dim=2)
print(TCR_antigen_result2.shape)
TCR_antigen_result2=TCR_antigen_result2.reshape(len(TCR_list2),37,-1)
TCR_antigen_result2=TCR_antigen_result2.to(device)

TCR_antigen_result_sum2=torch.cat((TCR_antigen_result2,TCR_antigen_result_blosum2),dim=1)
TCR_antigen_result_sum2=TCR_antigen_result_sum2.to(device)

print(TCR_antigen_result2.shape)
label2=pr.get_label("data/test.csv")  # bugfix: was the absolute path "/data/test.csv" (typo), see RESTORATION_NOTES.md
label2=torch.Tensor(label2)
label2=torch.unsqueeze(label2,1)
label2=label2.to(device)

dataset = TensorDataset(TCR_antigen_result_sum2,label2)
test_dataloader = DataLoader(dataset, batch_size=256)

# model_pre.eval()
test_loss = 0
correct = 0
total = 0
r2=0
Accuracy=0

fpr_arr=np.zeros(54)
tpr_arr=np.zeros(54)
precision_arr=np.zeros(54)
recall_arr=np.zeros(54)
with torch.no_grad():
    model_pre.eval()
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
        outputs = model_pre(inputs.data)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        try:
            roc_auc += roc_auc_score(np.array(labels.cpu()),np.array(predicted.cpu()))
            fpr_result, tpr_result, _ = roc_curve(np.array(labels.cpu()),np.array(predicted.cpu()))
            auc1 += auc(fpr_result, tpr_result)
            # 计算PR曲线和AUPR
            precision, recall, _ = precision_recall_curve(np.array(labels.cpu()),np.array(predicted.cpu()))
            aupr += average_precision_score(np.array(labels.cpu()),np.array(predicted.cpu()))
        except ValueError:
            pass
    print(auc1/53)
    auc_sum=roc_auc/53
    aupr_sum=aupr/53
    prAUC_sum=prAUC/53
    accuracy = correct / total
    print('AUC:',auc_sum)
    print('AUPR:',aupr_sum)
    print('prAUC_sum:',prAUC_sum)