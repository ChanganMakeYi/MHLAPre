import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
HLA_seq_lib={}
HLA_seq_list=[]
path='hla_library'
HLA_ABC=[path+'/A_prot.fasta',path+'/B_prot.fasta',path+'/C_prot.fasta',path+'/E_prot.fasta']

aa_dict_one_hot = {'A': 0,'C': 1,'D': 2,'E': 3,'F': 4,'G': 5,'H': 6,'I': 7,'K': 8,'L': 9,
           'M': 10,'N': 11,'P': 12,'Q': 13,'R': 14,'S': 15,'T': 16,'V': 17,
           'W': 18,'Y': 19}
amino_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
def load_list_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        lst = []
        for line in lines:
            lst.append(line)
    return lst

def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

for one_class in HLA_ABC:
    prot=open(one_class)
    #pseudo_seq from netMHCpan:https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796
    pseudo_seq_pos=[7,9,24,45,59,62,63,66,67,79,70,73,74,76,77,80,81,84,95,97,99,114,116,118,143,147,150,152,156,158,159,163,167,171]
    #write HLA sequences into a library
    #class I alles
    name=''
    sequence=''
    for line in prot:
        if len(name)!=0:
            if line.startswith('>HLA'):
                pseudo=''
                for i in range(0,34):
                    if len(sequence)>pseudo_seq_pos[i]:
                        pseudo=pseudo+sequence[pseudo_seq_pos[i]]
                HLA_seq_lib[name]=pseudo
                name=line.split(' ')[1]
                sequence=''
            else:
                sequence=sequence+line.strip()
        else:
            name=line.split(' ')[1]

def hla_encode(HLA_name):
    '''Convert the HLAs of a sample(s) to a zero-padded (for homozygotes)
    numeric representation.

    Parameters
    ----------
        HLA_name: the name of the HLA
        encoding_method:'BLOSUM50' or 'one-hot'
    '''
    if HLA_name not in HLA_seq_lib.keys():
        if len([hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))])==0:
            print('cannot find'+HLA_name)
        else:
            HLA_name=[hla_allele for hla_allele in HLA_seq_lib.keys() if hla_allele.startswith(str(HLA_name))][0]
    if HLA_name not in HLA_seq_lib.keys():
        print('Not proper HLA allele:'+HLA_name)
        return 0
    HLA_sequence=HLA_seq_lib[HLA_name]

    return HLA_sequence
# HLA_all=pd.read_csv('data/Table4.csv')
#
# MHCRestriction=pd.Series(list(HLA_all['MHCRestriction']))
# MHCRestriction=(MHCRestriction.dropna()).tolist()
# for HLA_name in tqdm(MHCRestriction):
#     HLA_sequence=hla_encode(HLA_name)
#     HLA_seq_list.append(HLA_sequence)
# # 保存列表到文件
# save_list_to_file(HLA_seq_list, 'my_list.txt')
# HLA_seq_list=load_list_from_file('my_list.txt')
#
# seq_zhuan_list=[0]*34
# # 创建20*34的全0数组
# arr = np.zeros((34*20))
# arr = arr.reshape((34, 20))
# print(len(HLA_seq_list))
# for i in range(34):
#     seq_zhuan = ''
#     for j in range(len(HLA_seq_list)-1):
#         seq_zhuan=seq_zhuan+HLA_seq_list[j][i]
#
#     seq_zhuan_list[i]=seq_zhuan
# print(len(seq_zhuan_list))
# # for j in range(20):
# #     print(j)
# #     print(amino_list[j])
# for i in range(34):
#     for j in range(20):
#         arr[i][j]=seq_zhuan_list[i].count(amino_list[j])
# arr=arr.astype(float)
# # arr = arr.reshape((20, 34))
# arr=arr/45112
# # 将数组转换为DataFrame对象
# df = pd.DataFrame(arr)
#
# # 将数据保存到Excel文件中
# # df.to_excel('data.xlsx', index=False)


HLA_list=[]
for key in HLA_seq_lib:
    if ((len(HLA_seq_lib[key])==34)&key.startswith('C')):
        HLA_list.append(HLA_seq_lib[key])
seq_zhuan_list=[0]*34
# 创建20*34的全0数组
arr = np.zeros((34*20))
arr = arr.reshape((34, 20))
for i in range(34):
    seq_zhuan = ''
    for j in range(len(HLA_list)-1):
        seq_zhuan=seq_zhuan+HLA_list[j][i]
    seq_zhuan_list[i]=seq_zhuan
print(len(seq_zhuan_list))
# for j in range(20):
#     print(j)
#     print(amino_list[j])
for i in range(34):
    for j in range(20):
        arr[i][j]=seq_zhuan_list[i].count(amino_list[j])
arr=arr.astype(float)
# arr = arr.reshape((20, 34))
arr=arr/len(HLA_list)

print(arr)
df = pd.DataFrame(arr)

# 将数据保存到Excel文件中
df.to_excel('data6.xlsx', index=False)