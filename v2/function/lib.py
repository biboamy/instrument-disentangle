import torch
import numpy as np
import torch.nn.init as init
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def get_weight(Ytr):
    mp = Ytr[:].sum(0).sum(1)
    mmp = mp.astype(np.float32) / mp.sum()
    cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
    cc[3]=1
    inverse_feq = torch.from_numpy(cc)
    return inverse_feq

class Data2Torch(Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.YI = data[1]
        self.YP = data[2]
        self.YS = data[3]

    def __getitem__(self, index):
        mX = torch.from_numpy(self.X[index]).float()
        mYI = torch.from_numpy(self.YI[index]).float()
        mYP = torch.from_numpy(self.YP[index]).float()
        mYS = torch.from_numpy(self.YS[index]).float()
        return mX, mYI, mYP, mYS
 
    def __len__(self):
        return len(self.X)

def loss_func(pred, tar, gwe, name, isAdv):

    we = gwe[0].cuda()
    wwe = 10
    we *= wwe
    
    loss = 0

    def inst_loss(inst_pred, inst_tar):
        loss_i = 0
        for idx, (out, fl_target) in enumerate(zip(inst_pred,inst_tar)):
            twe = we.view(-1,1).repeat(1,fl_target.size(1)).type(torch.cuda.FloatTensor)
            ttwe = twe * fl_target.data + (1 - fl_target.data) * 1
            loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
            loss_i += loss_fn(torch.squeeze(out), fl_target)
        
        return loss_i

    def pitch_loss(pit_pred, pit_tar):
        loss_p = 0
        for idx, (out, fl_target) in enumerate(zip(pit_pred,pit_tar)):
            ttwe = 10 * fl_target.data + (1 - fl_target.data) * 1
            loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
            loss_p += loss_fn(out, fl_target)
        return loss_p

    def stream_loss(str_pred, str_tar):

        loss_s = 0
        for idx, (out, fl_target) in enumerate(zip(str_pred,str_tar)):
            ttwe = 10 * fl_target.data + (1 - fl_target.data) * 1
            loss_fn = nn.BCEWithLogitsLoss(weight=ttwe, size_average=True)
            los = loss_fn(out, fl_target)
            loss_s += los
         
        return loss_s

    l0,l1,l2,l3,l4=torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)
    if 'Unet' in name:
        if not isAdv:
            l4 = stream_loss(pred[4],tar[2])*90
            if 'preIP' in name:
                l0 = inst_loss(pred[0],tar[0])
            if 'prePP' in name:
                l1 = pitch_loss(pred[1],tar[1])*9
        else:
            if 'prePNZ' in name:
                l3 = pitch_loss(pred[3],Variable(torch.zeros(pred[3].size())).cuda())*9
            if 'prePNN' in name:
                l3 = -pitch_loss(pred[3],tar[1])*9
    if 'Duo' in name:
        if 'preIP' in name and not isAdv:
            l0 = inst_loss(pred[0],tar[0])
        if 'prePP' in name and not isAdv:
            l1 = pitch_loss(pred[1],tar[1])*9
        if 'preINZ' in name and isAdv:
            l2 = inst_loss(pred[2],Variable(torch.zeros(pred[2].size())).cuda())
        if 'preINN' in name and isAdv:
            l2 = -inst_loss(pred[2],tar[0])
        if 'prePNZ' in name and isAdv:
            l3 = pitch_loss(pred[3],Variable(torch.zeros(pred[3].size())).cuda())*9
        if 'prePNN' in name and isAdv:
            l3 = -pitch_loss(pred[3],tar[1])*9
        if 'preRoll' in name and not isAdv:
            l4 = stream_loss(pred[4],tar[2])*90
    return l0,l1,l2,l3,l4
    
       
def model_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))