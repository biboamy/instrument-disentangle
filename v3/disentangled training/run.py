import datetime,os,torch
from torch.utils.data import Dataset
from loadData import *
from lib import *
from fit import *
from model import *
from skimage.measure import block_reduce
import sys
date = datetime.datetime.now()
os.environ['CUDA_VISIBLE_DEVICES'] = '6' # change

# UnetAE_preRoll
# UnetAE_preIP_preRoll 						-> only predict instrument at buttom
# UnetAE_preIP_prePP_prePNZ_preRoll 		-> with pitch as adv training predict zero
# UnetAE_preIP_prePP_prePNN_preRoll 		-> with pitch as adv training negative loss
# DuoAE_preIP_prePP 						-> no pianoroll conntected
# DuoAE_preIP_preINZ_prePP_prePNZ 			-> no pianoroll conntected and adv training predict zero
# DuoAE_preIP_preINN_prePP_prePNN 			-> no pianoroll conntected and adv training negative loss
# DuoAE_preIP_prePP_preRoll 				-> pianoroll conntected 
# DuoAE_preIP_preINZ_prePP_prePNZ_preRoll 	-> pianoroll conntected and adv training predict zero
# DuoAE_preIP_preINN_prePP_prePNN_preRoll 	-> pianoroll conntected and adv training predict negative

def main(args):
	name = args[1]
	batch_size = 10
	epoch = 100
	lr = 0.01

	out_model_fn = '../data/model/%d%d%d/%s/'%(date.year,date.month,date.day,name)
	if not os.path.exists(out_model_fn):
	    os.makedirs(out_model_fn)

	# load data
	t_kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True,'drop_last': True}
	Xtr,Ytr,Ytr_p,Ytr_s = load('musescore') 
	# Xtr (batch_size, 1, note_bin, time_length)
	# Ytr (batch_size, instrument_categories, time_length)
	# Ytr_p (batch_size, note_bin, time_length)
	# Ytr_s (batch_size, instrument_categories, note_bin, time_length)
	Xtr_mel,Ytr_mel = load_melody('musescore')
	avg, std = np.load('../../data/cqt_avg_std.npy')
	trdata = [Xtr, Ytr, Ytr_p, Ytr_s, Xtr_mel, Ytr_mel]
	tr_loader = torch.utils.data.DataLoader(Data2Torch(trdata), shuffle=True, **t_kwargs)
	print('finishing data building...')

	# build model
	model = Net(name).cuda()
	model.apply(model_init)

	# balance data
	weight = [get_weight(Ytr)]

	# start training
	Trer = Trainer(model, lr, epoch, out_model_fn, avg, std)
	Trer.fit(tr_loader, weight, name)

	print( out_model_fn)

if __name__ == "__main__":
    main(sys.argv)
