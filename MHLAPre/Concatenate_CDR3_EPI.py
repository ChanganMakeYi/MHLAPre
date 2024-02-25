import numpy as np
import torch
CDR3=np.load('ProcessData/CDR3.npy')
TCR_EPI=np.load('ProcessData/TCR_EPI.npy')

EPI_CDR3=np.concatenate((TCR_EPI,CDR3),axis=1)
print(EPI_CDR3.shape)
MHC_TCR_EP1=np.load('ProcessData/MHC_TCR_EP1.npy')
MHC_TCR_EP2=np.load('ProcessData/MHC_TCR_EP2.npy')
MHC_TCR_EP3=np.load('ProcessData/MHC_TCR_EP3.npy')
MHC_TCR_EP4=np.load('ProcessData/MHC_TCR_EP4.npy')

MHC=np.concatenate((MHC_TCR_EP1,MHC_TCR_EP2,MHC_TCR_EP3,MHC_TCR_EP4),axis=0)

EP_TCR_MHC=(torch.cat((torch.Tensor(EPI_CDR3),torch.Tensor(MHC)),dim=1)).reshape(-1,21,74)

