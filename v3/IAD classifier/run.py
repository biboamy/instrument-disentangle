import torch, os, h5py, datetime, sys, time, random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import SharedArray as sa
import numpy as np
import torch.nn.init as init
from torch.utils.data import Dataset
from torch.autograd import Variable
from skimage.measure import block_reduce
import torch.optim as optim
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
date = datetime.datetime.now()

batch_size = 16
lr = 0.008
epoch = 10 
nums_label = 10

# UnetAE_preIP_preRoll 
# UnetAE_preIP_prePP_prePNZ_preRoll 
# UnetAE_preIP_prePP_prePNN_preRoll 
# DuoAE_preIP_prePP_preRoll
# DuoAE_preIP_preINZ_prePP_prePNZ_preRoll
# DuoAE_preIP_preINN_prePP_prePNN_preRoll
# twoStep
# MTAN
name = 'DuoAE_preIP_preINZ_prePP_prePNZ_preRoll'

saveName = name
load_name = name + '_train' 
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

def load():
	label, data = [],[]
	for f in os.listdir('feature/latent_inst_'+str(nums_label)+'/'+load_name+'/'):
		label.append(np.load('feature/label_train_'+str(nums_label)+'/'+f))
		xi = np.load('feature/latent_inst_'+str(nums_label)+'/'+load_name+'/'+f)
		data.append(xi)
	label = np.concatenate(label,0)
	data = np.concatenate(data,0)
	return data, label

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

    def __getitem__(self, index):
        mX = torch.from_numpy(self.X[index]).float()
        mY = torch.from_numpy(self.Y[index]).float()
        return mX, mY
    
    def __len__(self):
        return len(self.X)

def loss_func(pred, tar, gwe):

    we = gwe[0].cuda()
    wwe = 10
    we *= wwe
    
    loss_i = 0
    for idx, (out, fl_target) in enumerate(zip(pred,tar)):
        twe = we.view(-1,1).repeat(1,fl_target.size(1)).type(torch.cuda.FloatTensor)
        ttwe = twe * fl_target.data + (1 - fl_target.data) * 1
        loss_fn = nn.BCEWithLogitsLoss(weight=ttwe)
        loss_i += loss_fn(torch.squeeze(out), fl_target)
        
    return loss_i

class Trainer:
    def __init__(self, model, lr, epoch, save_fn):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        print('Start Training #Epoch:%d'%(epoch))
    
    def fit(self, tr_loader, weight):
        st = time.time()
        save_dict = {}
        for e in range(1, self.epoch+1):
            lr = self.lr / (((e/(50)))+1)
            print( '\n==> Training Epoch #%d lr=%4f'%(e, lr))
            loss_total = 0
            for batch_idx, _input in enumerate(tr_loader):
                self.model.train()
                inp,tar = Variable(_input[0].cuda()), Variable(_input[1].cuda())
                predict = self.model(inp)
                opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                opt.zero_grad() 
                loss = loss_func(predict, tar, weight)
                loss_total+=loss
                loss.backward()
                opt.step()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d] loss %3f  Time %d'
                            %(e, self.epoch, batch_idx+1, len(tr_loader), loss.data, time.time() - st))
                sys.stdout.flush()
   			
            print ('\n')
            print(loss_total)
            save_dict['state_dict'] = self.model.state_dict()
            torch.save(save_dict, self.save_fn+'e_%d'%(e))

class conv_block(nn.Module):
    def __init__(self, inp, out, kernal, pad):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(inp, out, kernal, padding=pad)
        self.batch = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.batch(self.conv(x)))
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.head = nn.Sequential(
        	conv_block(128,128*2,(5,3),(0,1)),
        	conv_block(128*2,128*3,(1,3),(0,1)),
        	conv_block(128*3,128*3,(1,3),(0,1)),
        	conv_block(128*3,128,(1,3),(0,1))
        )

        self.head2 = nn.Sequential(
        	nn.Linear(128,1024),
        	nn.Linear(1024,nums_label)
        )
 
    def forward(self, _input):
    	oup = self.head(_input).squeeze(2)
    	
    	oup = self.head2(oup.permute(0,2,1))
    	oup = F.max_pool2d(oup.permute(0,2,1),(1,2),(1,2))

    	return oup

def main():
    t_kwargs = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': True,'drop_last': True}
    Xtr,Ytr = load()
    Xtr = np.pad(Xtr,((0,0),(0,0),(0,0),(0,1)),'constant',constant_values=0)
    Ytr = block_reduce(Ytr, block_size=(1, 1, 30), func=np.max)[:,:,:-1]
    tr_loader = torch.utils.data.DataLoader(Data2Torch([Xtr,Ytr]), shuffle=True, **t_kwargs)
    print('finishing loading data...')

    model = Net().cuda()
    model.apply(model_init)
    print('finishing loading model...')

    inst_weight = [get_weight(Ytr)]
    Trer = Trainer(model, lr, epoch, out_model_fn)
    Trer.fit(tr_loader,inst_weight)
main()