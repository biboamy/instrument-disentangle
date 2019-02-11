import torch, os, h5py, datetime, sys, time, random
import SharedArray as sa
from model import *
import numpy as np
import torch.nn.init as init
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.optim as optim
print(sa.list())
date = datetime.datetime.now()

batch_size = 8
lr = 0.001
epoch = 20 

saveName = 'train_coversong_exfeature'
out_model_fn = './model/%d%d%d/%s/'%(date.year,date.month,date.day,saveName)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

def get_weight(Ytr):
    mp = Ytr[:].sum(0).sum(1)
    mmp = mp.astype(np.float32) / mp.sum()
    cc=((mmp.mean() / mmp) * ((1-mmp)/(1 - mmp.mean())))**0.3
    cc[3]=1
    inverse_feq = torch.from_numpy(cc)
    return inverse_feq

def load(data_name):
    
    try:
        Xtr = sa.attach('shm://%s_Xtr'%(data_name))
        Ytr = sa.attach('shm://%s_Ytr'%(data_name))
        Ytr_p = sa.attach('shm://%s_Ytr_pitch'%(data_name))
        Ytr_s = sa.attach('shm://%s_Ytr_stream'%(data_name))
    except:
        # load cqt 
        trdata = h5py.File('../../ex_data/musescore500/tr.h5', 'r')
        Xtr = sa.create('shm://%s_Xtr'%(data_name), (trdata['x'].shape), dtype='float32')
        Xtr[:] = trdata['x'][:]
        #load instrument label
        Ytr = sa.create('shm://%s_Ytr'%(data_name), (trdata['y'].shape), dtype='float32')
        Ytr[:] = trdata['y'][:]
        #load pitch label
        trdata = h5py.File('../../ex_data/musescore500/tr_pitch.h5', 'r')
        Ytr_p = sa.create('shm://%s_Ytr_pitch'%(data_name), (trdata['y'].shape), dtype='float32')
        Ytr_p[:] = trdata['y'][:]
        #load pianoroll label
        trdata = h5py.File('../../ex_data/musescore500/tr_stream.h5', 'r')
        Ytr_s = sa.create('shm://%s_Ytr_stream'%(data_name), (trdata['y'].shape), dtype='float32')
        Ytr_s[:] = trdata['y'][:]
   
    return Xtr, Ytr, Ytr_p, Ytr_s

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

class Data2Torch(Dataset):
    def __init__(self, data):

        self.X = data[0]
        self.Y = data[1]
        self.YP = data[2]
        self.YS = data[3]

    def __getitem__(self, index):

        mX = torch.from_numpy(self.X[index]).float()
        mY = torch.from_numpy(self.Y[index]).float()
        mYP = torch.from_numpy(self.YP[index]).float()
        mYS = torch.from_numpy(self.YS[index]).float()
            
        return mX, mY, mYP, mYS
    
    def __len__(self):
        return len(self.X)

def loss_func(pred, tar, gwe):

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

    l1 = inst_loss(pred[0],tar[0])
    l2 = pitch_loss(pred[1], tar[1])
    l3 = inst_loss(pred[2],Variable(torch.zeros(pred[2].size())).cuda())
    l4 = pitch_loss(pred[3],Variable(torch.zeros(pred[3].size())).cuda())

    return l1, l2, l3, l4

class Trainer:
    def __init__(self, model, lr, epoch, save_fn, avg, std):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.Xavg, self.Xstd = Variable(torch.from_numpy(avg).cuda()), Variable(torch.from_numpy(std).cuda())

        print('Start Training #Epoch:%d'%(epoch))
    
    def fit(self, tr_loader, weight):
        st = time.time()
        save_dict = {}
        for e in range(1, self.epoch+1):
            print( '\n==> Training Epoch #%d lr=%4f'%(e, lr))
            for batch_idx, _input in enumerate(tr_loader):
                self.model.train()
                inp,tar = Variable(_input[0].cuda()), [Variable(_input[1].cuda()), Variable(_input[2].cuda()), Variable(_input[3].cuda())]
           
                predict = self.model(inp, self.Xavg, self.Xstd)

                opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                opt.zero_grad() 

                loss = loss_func(predict, tar,weight)
                sum(loss).backward()
                opt.step()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d] inst_pos %3f pitch_pos %3f inst_neg %3f pitch_neg %3f  Time %d'
                            %(e, self.epoch, batch_idx+1, len(tr_loader), loss[0].data, loss[1].data, loss[2].data, loss[3].data, time.time() - st))
                sys.stdout.flush()
   
            print ('\n')
            save_dict['state_dict'] = self.model.state_dict()
            torch.save(save_dict, self.save_fn+'e_%d'%(e))

def main():
    t_kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True,'drop_last': True}
    Xtr,Ytr,Ytr_p,Ytr_s = load('musescore')
    tr_loader = torch.utils.data.DataLoader(Data2Torch([Xtr,Ytr,Ytr_p,Ytr_s]), shuffle=True, **t_kwargs)
    print('finishing loading data...')

    model = Net().cuda()
    model.apply(model_init)
    print('finishing loading model...')

    avg, std = np.load('../../data/cqt_avg_std.npy')
    inst_weight = [get_weight(Ytr)]

	# start training
    Trer = Trainer(model, lr, epoch, out_model_fn, avg, std)
    Trer.fit(tr_loader,inst_weight)

main()