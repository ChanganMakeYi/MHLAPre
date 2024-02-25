import torch
import torch.nn as nn
import pandas as pd
import numpy as np

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

num_layers = 6
d_model = 50
nhead = 10
dim_feedforward = 2048
dropout = 0.1
HLA_all=pd.read_csv('data/Table3.csv')
epitope_map=torch.Tensor(np.load('ProcessData/epitope_map.npy'))
HLA_map1=torch.Tensor(np.load('ProcessData/HLA_map1.npy'))
HLA_map2=torch.Tensor(np.load('ProcessData/HLA_map2.npy'))
HLA_map3=torch.Tensor(np.load('ProcessData/HLA_map3.npy'))
HLA_map4=torch.Tensor(np.load('ProcessData/HLA_map4.npy'))
HLA_map=np.concatenate((HLA_map1,HLA_map2,HLA_map3,HLA_map4),axis=0)
input_data=torch.cat((epitope_map,torch.Tensor(HLA_map)),dim=1)

encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)


output = encoder(input_data)


