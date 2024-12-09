import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
# Sinusoidal position encoding
position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 50) for j in range(21)] for pos in range(50)])
position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
position_encoding = torch.from_numpy(position_encoding)


def add_position_encoding(seq):
    padding_ids = torch.abs(seq).sum(dim=-1) == 0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
    return seq

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.feedforward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

# #add position information
# epitope_map=torch.Tensor(np.load('ProcessData/epitope_map.npy'))
# HLA_map1=torch.Tensor(np.load('ProcessData/MHC_TCR_EP1.npy'))
# HLA_map3=torch.Tensor(np.load('ProcessData/MHC_TCR_EP2.npy'))
# HLA_map4=torch.Tensor(np.load('ProcessData/MHC_TCR_EP3.npy'))
# HLA_map=np.concatenate((HLA_map1,HLA_map3,HLA_map4),axis=0)
#
# # 创建一个新的形状为[45110, 50, 21]的零张量
# input_data = np.zeros((45110, 50, 21))
#
# # 将两个张量合并到新的张量中
# input_data[:, :16, :] = epitope_map[:, :, :]
# input_data[:, 16:50, :] = HLA_map
# input_data=torch.Tensor(input_data)

input_data=np.load('ProcessData/Transfer_data/hla_epit_cdr3.npy')
input_data=torch.Tensor(input_data)
print(input_data.shape)
#phla pre-train with position_encoder
position_input_data = torch.zeros(input_data.shape[0], 50, 21)
for i in tqdm(range(len(input_data))):
    position_data=add_position_encoding(input_data[i].reshape(50, 21))
    position_input_data[i]=position_data

input_data=position_input_data
print(input_data.shape)
# pre-train with transformer
num_layers = 3
d_model = 50
nhead = 10
dim_feedforward = 2048
dropout = 0.1
input_data=input_data.reshape(-1,21,50)
encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
# def transfomers_encoder_batch(input_data):
#     return_data = torch.Tensor(np.zeros((input_data.size(0), 50, 21)))
#     return_data=return_data
#     for i in tqdm(range(input_data.size(0))):
#         output = encoder(input_data[i].reshape(50,21))
#         return_data[i]=output
return_data=encoder(input_data)
print(return_data.shape)
return_data=return_data.detach().numpy()
np.save('ProcessData/Transfer_data/transfomer_data_mhc_ep_cdr3.npy',return_data)




