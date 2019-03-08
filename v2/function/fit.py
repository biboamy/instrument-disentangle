import torch, time, sys
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from lib import *

class Trainer:
    def __init__(self, model, lr, epoch, save_fn, avg, std):
        self.epoch = epoch
        self.model = model
        self.lr = lr
        self.save_fn = save_fn
        self.Xavg, self.Xstd = Variable(torch.from_numpy(avg).cuda()), Variable(torch.from_numpy(std).cuda())

        print('Start Training #Epoch:%d'%(epoch))
    
    def fit(self, tr_loader, weight, name):
        st = time.time()
        #save dict
        save_dict = {}

        for e in range(1, self.epoch+1):
            lr = self.lr / (((e//(70*1))*2)+1) 
            print( '\n==> Training Epoch #%d lr=%4f'%(e, lr))

            # Training
            for batch_idx, _input in enumerate(tr_loader):
                self.model.train()
                data, target = Variable(_input[0].cuda()), [Variable(_input[1].cuda()), Variable(_input[2].cuda()), Variable(_input[3].cuda())]
                
                opt = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                opt.zero_grad()

                predict = self.model(data, self.Xavg, self.Xstd, name, False)
                loss = loss_func(predict, target, weight, name, False)
                total_loss = [l for l in loss if (l>0).all()]
                sum(total_loss).backward()
                opt.step()

                loss_adv = loss
                if ('prePN' in name) or ('preIN' in name):
                    if 'UnetAE' in name: params = self.model.encode.parameters()
                    if 'DuoAE' in name: params = list(self.model.pitch_encode.parameters())+list(self.model.inst_encode.parameters())
                    opt_adv = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
                    opt_adv.zero_grad() 

                    predict_adv = self.model(data, self.Xavg, self.Xstd, name, True)
                    loss_adv = loss_func(predict_adv, target, weight, name, True)
                    total_loss_adv = [l for l in loss_adv if (l!=0).all()]
                    sum(total_loss_adv).backward()
                    opt_adv.step()
                
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d] Inst:%3f  Pitch:%3f  Inst-N:%3f  Pitch-N:%3f  Roll:%3f  Time %d'
                            %(e, self.epoch, batch_idx+1, len(tr_loader), loss[0].data, loss[1].data, loss_adv[2].data,loss_adv[3].data,loss[4].data, time.time() - st))
                sys.stdout.flush()
            print ('\n')
            save_dict['state_dict'] = self.model.state_dict()
            torch.save(save_dict, self.save_fn+'e_%d'%(e))