from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import Pretreatment as pr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNModule(nn.Module):
    def __init__(self):
        super(CNNModule, self).__init__()
        self.fc1 = nn.Linear(925, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Reconstructed (not recovered from git history, see RESTORATION_NOTES.md).
# CNNModule's fc1 takes 925 = 37 * 25: the flattened per-residue feature dim
# (5 Atchley factors + 32 embedding_32.txt dims) times the TCR CDR3 encode
# length used everywhere else in this file (25). CNNModule2 is its antigen/
# epitope-side twin, so its input is 555 = 37 * 15 (the antigen encode length
# used below is 15).
class CNNModule2(nn.Module):
    def __init__(self):
        super(CNNModule2, self).__init__()
        self.fc1 = nn.Linear(555, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Reconstructed module-level preprocessing (not recovered from git history).
# main.py does `import DataPre as dp` and then immediately reads
# `dp.TCR_antigen_result` / `dp.TCR_antigen_result_sum` before it (redundantly)
# recomputes the same two tensors itself a few lines later from the same
# "data/train.csv" — so this file must build them at import time, exactly the
# way main.py's own local copy of this block does.
f = open("encode/embedding_32.txt", "r")
lines = f.readlines()
for line in lines[1:21]:
    li = line.split(',')
    myarray = np.array(li[1:]).astype(float)
    # pr.aa_dict_atchley is a module-level dict shared with every importer of
    # Pretreatment (including main.py, which redoes this same concatenation
    # right after `import DataPre as dp`). Guard against extending an
    # already-extended vector a second time (5 -> 37 -> 69), which silently
    # breaks every fixed dimension (37, 925, 555, ...) derived from it.
    if len(pr.aa_dict_atchley[li[0]]) == 5:
        pr.aa_dict_atchley[li[0]] = np.concatenate((pr.aa_dict_atchley[li[0]], myarray), axis=0)
f.close()
torch.set_default_dtype(torch.float64)

TCR_list, antigen_list = pr.preprocess("data/train.csv")

antigen_array_blosum = torch.Tensor(pr.antigenMap(antigen_list, 15, 'BLOSUM50'))
TCR_array_blosum = torch.Tensor(pr.antigenMap(TCR_list, 25, 'BLOSUM50'))
TCR_antigen_result_blosum_ori = torch.cat((antigen_array_blosum, TCR_array_blosum), dim=1)
TCR_antigen_result_blosum = TCR_antigen_result_blosum_ori.reshape(len(antigen_list), 21, -1)

antigen_array = torch.Tensor(pr.aamapping_TCR(antigen_list, pr.aa_dict_atchley, 15))
antigen_array = antigen_array.reshape(len(antigen_list), 37, -1)

TCR_array = torch.Tensor(pr.aamapping_TCR(TCR_list, pr.aa_dict_atchley, 25))
TCR_array = TCR_array.reshape(len(TCR_list), 37, -1)

TCR_antigen_result = torch.cat((antigen_array, TCR_array), dim=2)
TCR_antigen_result_sum = torch.cat((TCR_antigen_result, TCR_antigen_result_blosum), dim=1)