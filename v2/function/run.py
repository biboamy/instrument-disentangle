import datetime,os,torch
from torch.utils.data import Dataset
from loadData import *
from lib import *
from fit import *
from model import *
date = datetime.datetime.now()
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # change

# some name to choose

# UnetAE_preRoll 							-> Unet no disentanglement 
# UnetAE_preIP_preRoll 						-> Unet with disentanglement no adversarial training
# UnetAE_preIP_prePP_prePNZ_preRoll 		-> Unet with disentanglement and adversarial training (zero target)
# UnetAE_preIP_prePP_prePNN_preRoll 		-> Unet with disentanglement and adversarial training (neg loss)
# DuoAE_preIP_prePP 						-> Duo no pianoroll no disentanglement
# DuoAE_preIP_preINZ_prePP_prePNZ 			-> Duo no pianoroll with disentanglement (zero target)
# DuoAE_preIP_preINN_prePP_prePNN 			-> Duo no pianoroll with disentanglement (neg loss)
# DuoAE_preIP_prePP_preRoll 				-> Duo with pianoroll no disentanglement
# DuoAE_preIP_preINZ_prePP_prePNZ_preRoll	-> Duo with pianoroll with disentanglement (zero target)
# DuoAE_preIP_preINN_prePP_prePNN_preRoll	-> Duo with pianoroll with disentanglement (neg loss)

name = 'DuoAE_preIP_preINN_prePP_prePNN_preRoll'
batch_size = 10
epoch = 100
lr = 0.01

out_model_fn = '../data/model/%d%d%d/%s/'%(date.year,date.month,date.day,name)
if not os.path.exists(out_model_fn):
    os.makedirs(out_model_fn)

# load data
t_kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True,'drop_last': True}
Xtr,Ytr,Ytr_p,Ytr_s = load('musescore')
avg, std = np.load('../data/cqt_avg_std.npy')
trdata = [Xtr, Ytr, Ytr_p, Ytr_s]
tr_loader = torch.utils.data.DataLoader(Data2Torch(trdata), shuffle=True, **t_kwargs)
print('finishing data building...')

# build model
model = Net(name).cuda()
model.apply(model_init)

# calculate instrument weight
weight = [get_weight(Ytr)]

# start training
Trer = Trainer(model, lr, epoch, out_model_fn, avg, std)
Trer.fit(tr_loader, weight, name)

print( out_model_fn)
