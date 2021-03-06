import torch.nn as nn
import torch,sys,math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from structure import *

batchNorm_momentum = 0.1

class block(nn.Module):
    def __init__(self,inp,out):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(inp,out, kernel_size=ENCODER['res_block']['conv']['ksize'], padding=ENCODER['res_block']['conv']['padding'])
        self.bn1 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.conv2 = nn.Conv2d(out, out, kernel_size=ENCODER['res_block']['conv']['ksize'], padding=ENCODER['res_block']['conv']['padding'])
        self.bn2 = nn.BatchNorm2d(out, momentum= batchNorm_momentum)
        self.skip = nn.Conv2d(inp, out, kernel_size=1, padding=0)

    def forward(self, x):
        x11 = F.relu(self.bn1(self.conv1(x)))
        x12 = F.relu(self.bn2(self.conv2(x11)))
        x12 += self.skip(x)
        xp, idx = F.max_pool2d(x12,kernel_size=ENCODER['res_block']['pooling']['ksize'], \
                               stride=ENCODER['res_block']['pooling']['stride'], \
                               padding=ENCODER['res_block']['pooling']['padding'],\
                               return_indices=ENCODER['res_block']['pooling']['return_indices'])
        return xp, idx, x12.size()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.bn = nn.BatchNorm2d(1, momentum= batchNorm_momentum)
        self.layer1 = block(ENCODER['layer1']['inp'],ENCODER['layer1']['oup'])
        self.layer2 = block(ENCODER['layer2']['inp'],ENCODER['layer2']['oup'])
        self.layer3 = block(ENCODER['layer3']['inp'],ENCODER['layer3']['oup'])
        self.layer4 = block(ENCODER['layer4']['inp'],ENCODER['layer4']['oup'])
        self.gru = nn.GRU(ENCODER['gru']['inp'], ENCODER['gru']['oup'], ENCODER['gru']['stack_num'], \
                          batch_first=ENCODER['gru']['batch_first'], bidirectional=ENCODER['gru']['bidirectional'])
        
    def forward(self,x):
        x = self.bn(x)
        x,idx,size = self.layer1(x)
        x,idx,size = self.layer2(x)
        x,idx,size = self.layer3(x)
        x,idx,size = self.layer4(x)
        _s = x.size()
        x,_ = self.gru(torch.transpose(x,1,3).contiguous().view(_s[0],312,-1))
        
        return torch.transpose(x.contiguous().view(_s[0],312,2,_s[1]),1,3)

class PitchDecoder(nn.Module):
    def __init__(self):
        super(PitchDecoder, self).__init__()
        
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3,1), stride=(3,1), padding=(1,0)),
            nn.BatchNorm2d(128, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(128, 64, kernel_size=(3,1), stride=(3,1), padding=(1,0)),
            nn.BatchNorm2d(64, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,1), stride=(3,1), padding=(0,0)),
            nn.BatchNorm2d(32, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(32, 1, kernel_size=(3,1), stride=(3,1), padding=(1,0))
        )
        
    def forward(self,x):
        x = self.decode(x)
        x = F.max_pool2d(x,(1,2),(1,2)).squeeze()
        return x

class InstDecoder(nn.Module):
    def __init__(self):
        super(InstDecoder, self).__init__()
        
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.BatchNorm2d(128, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(128, 64, kernel_size=(3,1), stride=(3,1), padding=(1,0)),
            nn.BatchNorm2d(64, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(64, 32, kernel_size=(3,1), stride=(1,1), padding=(1,0)),
            nn.BatchNorm2d(32, momentum= batchNorm_momentum),
            nn.ConvTranspose2d(32, 1, kernel_size=(3,1), stride=(3,1), padding=(1,0))
        )
    
    def forward(self,x):
        x = self.decode(x)
        x = F.max_pool2d(x,(1,2),(1,2)).squeeze()
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.PitchEncode = Encoder()
        self.InstEncode = Encoder()
        self.PitchDecode = PitchDecoder()
        self.InstDecode = InstDecoder()
        
    def forward(self, _input, Xavg, Xstd):

        predict = _input

        def get_inst_x(x,avg,std):
            xs = x.size()
            avg = avg.view(1, avg.size()[0],1,1).repeat(xs[0], 1, xs[2], 1).type('torch.cuda.FloatTensor')
            std = std.view(1, std.size()[0],1,1).repeat(xs[0], 1, xs[2], 1).type('torch.cuda.FloatTensor')
            x = (x - avg)/std
            return x
    
        x = _input.unsqueeze(3) 
        x = get_inst_x(x,Xavg,Xstd)
   
        x = x.permute(0,3,1,2)
        pitch_enc = self.PitchEncode(x)
        inst_enc = self.InstEncode(x)
     
        pitch_true = self.PitchDecode(pitch_enc)
        inst_true = self.InstDecode(inst_enc)

        pitch_false = self.PitchDecode(inst_enc)
        inst_false = self.InstDecode(pitch_enc)

        predict = [inst_true,pitch_true,inst_false,pitch_false]
    
        return predict
