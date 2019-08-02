import librosa, torch
#from pypianoroll import Multitrack, Track
import numpy as np
import torch.nn.init as init
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from random import randint

def griffin_lim(mag_spec, n_fft, hop, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.
    Args:
        mag_spec (2D numpy array):  The magnitude spectrogram.
                                    (row: frame, col: frequency)
        n_fft (int):    The FFT size, which should be a power of 2.
        hop (int):      The hope size in samples.
        iterations (int):   Number of iterations for the Griffin-Lim algorithm.
                            (typically a few hundred is sufficien)
    Returns:
        The reconstructed time domain signal as a 1D numpy array.
    """

    # Reconstruct the signal for the "size/length" of desired audio signal.
    time_sample = librosa.istft(mag_spec, hop_length=hop,
                                win_length=n_fft, window='hanning')

    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(time_sample.shape[0])

    n = iterations  # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1

        reconstruction_spectrogram = librosa.stft(x_reconstruct, n_fft=n_fft,
                                                  hop_length=hop, window='hanning')
   
        reconstruction_angle = np.angle(reconstruction_spectrogram)

        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = mag_spec * np.exp(1.0j * reconstruction_angle)

        prev_x = x_reconstruct

        x_reconstruct = librosa.istft(proposal_spectrogram, hop_length=hop,
                                      win_length=n_fft, window='hanning')

        diff = sqrt(sum((x_reconstruct - prev_x)**2) / x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct

def write_midi(filepath, pianorolls, program_nums=None, is_drums=None,
               track_names=None, velocity=100, tempo=40.0, beat_resolution=24):
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    if not np.issubdtype(pianorolls.dtype, np.bool_):
        raise TypeError("Support only binary-valued piano-rolls")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]
    if pianorolls.shape[2] != len(program_nums):
        raise ValueError("`pianorolls` and `program_nums` must have the same"
                         "length")
    if pianorolls.shape[2] != len(is_drums):
        raise ValueError("`pianorolls` and `is_drums` must have the same"
                         "length")
    if program_nums is None:
        program_nums = [0] * len(pianorolls)
    if is_drums is None:
        is_drums = [False] * len(pianorolls)

    multitrack = Multitrack(beat_resolution=beat_resolution, tempo=tempo)
    for idx in range(pianorolls.shape[2]):
        #plt.subplot(10,1,idx+1)
        #plt.imshow(pianorolls[..., idx].T,cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
        if track_names is None:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx])
        else:
            track = Track(pianorolls[..., idx], program_nums[idx],
                          is_drums[idx], track_names[idx])
        multitrack.append_track(track)
    #plt.savefig(cf.MP3Name)
    multitrack.write(filepath)

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
        self.XM = data[4] 

    def __getitem__(self, index):
        rint = randint(0, len(self.XM)-1)

        mX = torch.from_numpy(self.X[index]+self.XM[rint]).float()
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
            l3 = pitch_loss(pred[3],Variable(torch.zeros(pred[3].size())).cuda())*9
    if 'Duo' in name:
        if 'preIP' in name and not isAdv:
            l0 = inst_loss(pred[0],F.max_pool1d(tar[0],2))
        if 'prePP' in name and not isAdv:
            l1 = pitch_loss(pred[1],F.max_pool1d(tar[1],2))
        if 'preINZ' in name and isAdv:
            l2 = inst_loss(pred[2],Variable(torch.zeros(pred[2].size())).cuda())*9
        if 'prePNZ' in name and isAdv:
            l3 = pitch_loss(pred[3],Variable(torch.zeros(pred[3].size())).cuda())*9
        if 'preRoll' in name and not isAdv:
            l4 = stream_loss(pred[4],F.max_pool2d(tar[2],(1,2)))*90

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