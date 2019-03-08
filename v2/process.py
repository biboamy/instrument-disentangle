import os, sys, time
import SharedArray as sa
from pydub import AudioSegment
from bs4 import BeautifulSoup
from scipy.io.wavfile import write
import xml.etree.ElementTree
import librosa
import pretty_midi
from joblib import Parallel, delayed
import numpy as np
import sys
from norm_lib import *
import random
import h5py
import shutil

def get_id(num):
	if 1 <= num <= 4: return [num,0] #piano
	elif 25 <= num <= 26: return [num,1] #a_guitar
	elif 27 <= num <= 28: return [num,2] #e_guitar_clean
	elif 30 <= num <= 31: return [num,3] #e_guitar_dist
	elif num == 0: return [num,4] #drum
	elif 81 <= num <= 104: return [num,5] #synth
	elif 17 <= num <= 21: return [num,6] #organ
	#elif 57 == num: return [num,5] #trumpet
	#elif 66 == num: return [num,6] #sax
	elif 34 <= num <= 35: return [num,7] #e_bass
	elif 44 == num: return [num,8] #doublebass
	elif 41 == num: return [num,9] #violin
	elif 43 == num: return [num,10] #cello
	elif 74 == num: return [num, 11] #flute
	else: return None 
'''
1. split training, evaluate, test set and select the songs to balace each instrument number
'''
def split_test_train():
	s_path = 'musescore data midi path'
	f_list = os.listdir(s_path)
	f_list = random.sample(f_list, len(f_list))

	def get_name(_list,n_list, total, num):
		for i,f in enumerate(_list):
			try:
				midi_data = pretty_midi.PrettyMIDI(s_path+f)
				inst_num = np.zeros(12)
				for inst in midi_data.instruments:
					if inst.is_drum: key = '0'
					else: key = str(inst.program+1)
					_, _id = get_id(int(key))
					inst_num[_id] = 1
				tmp = total + inst_num
					
				if len(np.where( tmp > num )[0]) < 1 and len(np.where(inst_num>0)[0])>1:
					total = tmp
					n_list.append(f)
					print (str(i), str(total), len(n_list))
				if total.sum() == num*12: break
			except Exception as e: print(e)
		return n_list
	
	tename = get_name(f_list[:int(len(f_list)*0.1)],[],np.zeros(10),50)
	np.save('tename.npy',tename)
	vaname = get_name(f_list[int(len(f_list)*0.1):int(len(f_list)*0.2)],[],np.zeros(10),50)
	np.save('vaname.npy',vaname)
	trname = get_name(f_list[int(len(f_list)*0.2):],[],np.zeros(10),1500)
	np.save('trname.npy',trname)
#split_test_train()

def _extract_feature(f, s_path, d_path, i):
	def logCQT(file,h):
		sr = 16000
		y, sr = librosa.load(file,sr=sr)
		cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5*float(h), n_bins=88, bins_per_octave=12)
		return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

	if not (os.path.isfile(d_path+f.replace('mid', 'npy'))):
		try:
			cqt = logCQT(s_path+f.replace('mid', 'mp3'),1)
			np.save(d_path+f.replace('mid', 'npy'),phase)
		
		except Exception as e: print (e)
	else: print ('exist')


'''
2. Extract CQT feature
'''			
def extract_feature():
	s_path = 'musescore data mp3 path'
	d_path = 'feature/cqt/'
	l = np.load('tename.npy')[:]
	l = np.concatenate((l, np.load('vaname.npy')[:]), axis=0)
	l = np.concatenate((l, np.load('tename.npy')[:]), axis=0)
	if not os.path.exists(d_path):
		os.makedirs(d_path)
	Parallel(n_jobs=10)(delayed(_extract_feature)(f, s_path, d_path, i) for i,f in enumerate(l[:]))
#extract_feature()

def _parse_midi(f,i,s_path, choose): 
	if choose == 'inst':d_path = 'feature/inst/'
	if choose == 'pitch':d_path = 'feature/pitch/'
	sr = 16000
	f = f.replace('npz', 'mid')
	if not (os.path.isfile(d_path+f.replace('mid', 'npz'))):
		print(i)
		try:
			output_array = {}
			midi_data = pretty_midi.PrettyMIDI(s_path+f)
			tmp = np.load('feature/cqt/'+f.replace('mid','npy'))
			length = tmp.shape[1]
			for j,inst in enumerate(midi_data.instruments):
				if choose == 'inst':
					label = np.sum(inst.get_piano_roll(sr/512), axis=0)	
				else:
					label = inst.get_piano_roll(sr/512)
				
				if inst.is_drum: key = '0'
				else: key = str(inst.program+1)
				if key in output_array:
					if choose == 'inst':  
						output_array[key][:len(label)] = output_array[key][:len(label)] + label
					elif choose == 'pitch': 
						output_array[key][:,:label.shape[1]] = output_array[key][:,:label.shape[1]] + label
					output_array[key][output_array[key]>1] = 1
				else:
					if choose == 'inst':  
						output_array[key] = np.zeros(length)
						output_array[key][:len(label)] = label
					elif choose == 'pitch': 
						output_array[key] = np.zeros((128,length))
						output_array[key][:,:label.shape[1]] = label
				label[label>0] = 1
			if not os.path.exists(d_path):
				os.makedirs(d_path)
			np.savez(d_path+f.replace('mid', 'npz'), **output_array)
		except Exception as e: print(e)
	else:print('exist' )

'''
3. Get instrument and pitch labels
'''
def parse_midi():
	s_path = 'musescore data midi path'
	l = np.load("trname.npy")
	l = np.concatenate((l, np.load('vaname.npy')[:]), axis=0)
	l = np.concatenate((l, np.load('tename.npy')[:]), axis=0)
	choose = 'pitch' #inst/pitch
	Parallel(n_jobs=50)(delayed(_parse_midi)(f,i,s_path,choose) for i,f in enumerate(l[:1]))
#parse_midi()

'''
4. Build experiment data 
'''
def get_ex():
	def chunk(inp):
		chunk_size = 312
		x = [] #change
		inp = inp[:,:int(inp.shape[1]//chunk_size)*chunk_size]
		for i in range(int(inp.shape[1]//chunk_size)):
			data=inp[:,i*chunk_size:i*chunk_size+chunk_size]
			x.append(data)
		x = np.array(x)	
		return x

	def _get_ex(n_list,choose):

		x_stack = []
		yi_stack = []
		yp_stack = []
		ys_stack = []
		name = []
		for i,n in enumerate(n_list[:]): 
			print (i)

			try:
				labels_i = np.load('./feature/inst/'+n.replace('mid','npz'))
				labels_p = np.load('./feature/pitch/'+n.replace('mid','npz'))
				c_labels = [get_id(int(l)) for l in labels_p]
				if None in c_labels: continue
				x = np.load('feature/cqt/'+n.replace('mid','npy'))
				y_i = np.zeros((12, x.shape[1]))
				y_p = np.zeros((88, x.shape[1]))
				y_s = np.zeros((12, 88, x.shape[1]))
			
				for l in c_labels:
					inst_data = labels_i[str(l[0])]
					y_i[l[1]] = y_i[l[1]] + inst_data
					pitch_data = labels_p[str(l[0])][21:109]
					if l[0] != 0:
						y_p = y_p + pitch_data
						y_s[l[1]] = y_s[l[1]] + pitch_data
				
				c_x = chunk(x)
				c_y_i = chunk(y_i) 
				c_y_p = chunk(y_p) 
				c_y_s = chunk(y_s.reshape((-1, y_s.shape[2]))).reshape((-1,12,88,312))
				
				for j,(x,yi,yp,ys) in enumerate(zip(c_x,c_y_i,c_y_p,c_y_s)):
					if not np.all(yi==0):
						x_stack.append([x])
						yi_stack.append([yi])
						yp_stack.append([yp])
						ys_stack.append([ys])
						
				name.append(n)
			except Exception as e: print (e)
		
		return np.vstack(x_stack),np.vstack(yi_stack),np.vstack(yp_stack),np.vstack(ys_stack)#,np.vstack(xstft_stack)

	def create_h5(x,yi,yp,ys,name):
		hf_i = h5py.File('./data/'+name+'.h5', 'a')
		
		try:
			del hf_i['x']
			del hf_i['yi']
			del hf_i['yp']
			del hf_i['xs']
			
		except: pass
		
		hf_i.create_dataset('x', data=x, maxshape=(None, x.shape[1], x.shape[2]), chunks=True)
		hf_i.create_dataset('yi', data=yi, maxshape=(None, yi.shape[1], yi.shape[2]), chunks=True)
		hf_i.create_dataset('yp', data=yp, maxshape=(None, yp.shape[1], yp.shape[2]), chunks=True)
		hf_i.create_dataset('ys', data=ys, maxshape=(None, ys.shape[1], ys.shape[2], ys.shape[3]), chunks=True)
		print (x.shape, yi.shape, yp.shape, ys.shape)

	choose = 'tr' #tr,va,te
	x, yi, yp, ys = _get_ex(np.load(choose+'name.npy'),choose)
	create_h5(x, yi, yp, ys, choose)
#get_ex()

def RoW_norm(data, fn, fqs=128):

    if True:
        st = time.time()
        print ('Get std and average')
        
        common_sum = 0
        square_sum = 0

        # remove zero padding
        fle = data.shape[2]
        tfle = 0
        for i in range(len(data)):
            tfle += (data[i].sum(-1).sum(0)!=0).astype('int').sum()

            common_sum += data[i].sum(-1).sum(-1)
            square_sum += (data[i]**2).sum(-1).sum(-1)
            print (i)
            
        common_avg = common_sum / tfle
        square_avg = square_sum / tfle
        
        std = np.sqrt( square_avg - common_avg**2 )
        np.save(fn, [common_avg, std])
        print (time.time() - st)
    
        return common_avg, std

'''
6. Normalize the data
'''
def get_avg_std():
	trdata = h5py.File('./data/tr.h5', 'r')
    Xtr = sa.create('shm://%s_Xtr'%(data_name), (trdata['x'].shape), dtype='float32')
	avg, std = RoW_norm(np.expand_dims(Xtr, axis=3),'data/cqt_avg_std.npy')
	print (avg.shape)
	print (std.shape)
#get_avg_std()
