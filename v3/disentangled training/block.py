import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

batchNorm_momentum = 0.1
num_labels = 10

class block(nn.Module):
    def __init__(self, inp, out, ksize, pad, ds_ksize, ds_stride):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ksize, padding=pad)
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ksize, padding=pad)
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)
        self.ds = nn.Conv2d(out, out, kernel_size=ds_ksize, stride=ds_stride, padding=0)

    def forward(self, x):
        x11 = F.leaky_relu(self.bn1(self.conv1(x)))
        x12 = F.leaky_relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp = self.ds(x12)
        return xp, xp, x12.size()

class d_block(nn.Module):
    def __init__(self, inp, out, isLast, ksize, pad, ds_ksize, ds_stride, name):
        super(d_block, self).__init__()
        self.conv2d = nn.ConvTranspose2d(inp, int(inp/2), kernel_size=ksize, padding=pad)
        self.bn2d = nn.BatchNorm2d(int(inp/2), momentum= batchNorm_momentum)
        self.conv1d = nn.ConvTranspose2d(int(inp/2), out, kernel_size=ksize, padding=pad)
        
        if not isLast: 
            self.bn1d = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
            if 'UnetAE' in name:
                self.us = nn.ConvTranspose2d(inp-out, inp-out, kernel_size=ds_ksize, stride=ds_stride) 
            if 'DuoAE' in name:
                self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride) 
        else: 
            self.us = nn.ConvTranspose2d(inp, inp, kernel_size=ds_ksize, stride=ds_stride) 

    def forward(self, x, name, size=None, isLast=None, skip=None):
        x = self.us(x,output_size=size)
        if 'UnetAE' in name:
            if not isLast: x = torch.cat((x, skip), 1) 
        x = F.leaky_relu(self.bn2d(self.conv2d(x)))
        if isLast: x = self.conv1d(x)
        else:  x = F.leaky_relu(self.bn1d(self.conv1d(x)))
        return x

class Encode(nn.Module):
    def __init__(self,name):
        super(Encode, self).__init__()

        self.block1 = block(1,16,(3,3),(1,1),(2,2),(2,2))
        self.block2 = block(16,32,(3,3),(1,1),(2,2),(2,2))
        self.block3 = block(32,64,(3,3),(1,1),(2,2),(2,2))
        self.block4 = block(64,128,(3,3),(1,1),(2,2),(2,2))

        if 'UnetAE' in name:
            self.conv1 = nn.Conv2d(64,64, kernel_size=(3,3), padding=(1,1)) 
            self.conv2 = nn.Conv2d(32,32, kernel_size=(3,3), padding=(1,1)) 
            self.conv3 = nn.Conv2d(16,16, kernel_size=(3,3), padding=(1,1)) 

    def forward(self, x,name):
        x1,idx1,s1 = self.block1(x)
        x2,idx2,s2 = self.block2(x1)
        x3,idx3,s3 = self.block3(x2)
        x4,idx4,s4 = self.block4(x3)
       
        if 'UnetAE' in name:
            c1=self.conv1(x3) 
            c2=self.conv2(x2) 
            c3=self.conv3(x1) 
            return x4,[s1,s2,s3,s4],[c1,c2,c3,x1]
        else:
            return x4,[s1,s2,s3,s4]
        

class Decode(nn.Module):
    def __init__(self,name):
        super(Decode, self).__init__()
        if 'UnetAE' in name:
            self.d_block1 = d_block(192,64,False,(3,3),(1,1),(2,2),(2,2),name)
            self.d_block2 = d_block(96,32,False,(3,3),(1,1),(2,2),(2,2),name)
            self.d_block3 = d_block(48,16,False,(3,3),(1,1),(2,2),(2,2),name)
            self.d_block4 = d_block(16,num_labels,True,(3,3),(1,1),(2,2),(2,2),name)
        if 'DuoAE' in name:
            self.d_block1 = d_block(256,128,False,(3,3),(1,1),(2,2),(2,2),name)
            self.d_block2 = d_block(128,64,False,(3,3),(1,1),(2,2),(2,2),name)
            self.d_block3 = d_block(64,32,False,(3,3),(1,1),(2,2),(2,2),name)
            self.d_block4 = d_block(32,num_labels,True,(3,3),(1,1),(2,2),(2,2),name)

    def forward(self, x,name, s, c=[None,None,None,None]):

        x = self.d_block1(x,name,s[3],False,c[0])
        x = self.d_block2(x,name,s[2],False,c[1])
        x = self.d_block3(x,name,s[1],False,c[2])
        pred_ori = self.d_block4(x,name,s[0],True,c[3])
        pred = F.max_pool2d(pred_ori,(1,2),(1,2)).squeeze()
       
        return pred

class PitchDecoder(nn.Module):
    def __init__(self):
        super(PitchDecoder, self).__init__()

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=(3,3), padding=(0,2)),
            nn.BatchNorm2d(64, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=(3,3), padding=(0,1)),
            nn.BatchNorm2d(32, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(32, 1, kernel_size=(2,2), stride=(2,2), padding=(1,1))
        )
        
    def forward(self,x):
        ori_x = self.decode(x)
        x = F.max_pool2d(ori_x,(1,2),(1,2)).squeeze()
        return x

class InstPred(nn.Module):
    def __init__(self):
        super(InstPred, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(2660,512),
            nn.Linear(512,num_labels)
        )
        self.decode = nn.Sequential(
            nn.Linear(num_labels,512),
            nn.Linear(512,665)
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(7,14, kernel_size=(3,3), padding=(1,1)),
            nn.Conv2d(14,28, kernel_size=(3,3), padding=(1,1))
        )
    def forward(self,x):
        s = x.size()
        x = x.view(-1,s[1]*s[2]*s[3])
        pred = self.encode(x)
        inp = F.sigmoid(pred)

        choice = np.random.choice(10, 2)
        inp_mo = np.zeros(inp.size())
        inp_mo[:,choice[0]] = 1
        inp_mo[:,choice[1]] = 1
        inp_mo = Variable(torch.from_numpy(inp_mo).float().cuda())

        # original
        oup = self.decode(inp)
        oup = oup.view(-1,7,s[2],s[3])
        oup = self.decode2(oup)

        #modified
        oup_mo = self.decode(inp_mo)
        oup_mo = oup_mo.view(-1,7,s[2],s[3])
        oup_mo = self.decode2(oup_mo)

        return inp, oup, inp_mo, oup_mo

class InstDecoder(nn.Module):
    def __init__(self):
        super(InstDecoder, self).__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(2,3), stride=(2,3), padding=(0,2)),
            nn.BatchNorm2d(64, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(64, 32, kernel_size=(1,3), stride=(1,3), padding=(0,1)),
            nn.BatchNorm2d(32, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(32, 1, kernel_size=(1,2), stride=(1,2), padding=(0,1))
        )
        
    def forward(self,x):
        x = self.decode(x)
        x_max = F.max_pool2d(x,(1,2),(1,2)).squeeze()

        return x_max.squeeze()
