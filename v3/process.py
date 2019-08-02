import os, sys, librosa, pretty_midi, h5py
import SharedArray as sa
from joblib import Parallel, delayed
import numpy as np

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

def _extract_feature(f, s_path, d_path, i):
	def logCQT(file,h):
		sr = 16000
		y, sr = librosa.load(file,sr=sr)
		cqt = librosa.cqt(y, sr=sr, hop_length=512, fmin=27.5*float(h), n_bins=88, bins_per_octave=12)
		return ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0

	if not (os.path.isfile(d_path+f.replace('mp3', 'npy'))):
		try:
			cqt = logCQT(s_path+f,1)
			np.save(d_path+f.replace('mp3', 'npy'),phase)
		
		except Exception as e: print (e)
	else: print ('exist')


'''
1. Extract CQT feature
'''			
def extract_feature():
	s_path = '../database2/musescore/data/mp3/'
	d_path = 'feature/cqt/'
	l = os.listdir(s_path)
	if not os.path.exists(d_path):
		os.makedirs(d_path)
	Parallel(n_jobs=10)(delayed(_extract_feature)(f, s_path, d_path, i) for i,f in enumerate(l[:]))
#extract_feature()

def _parse_midi(f,i,s_path, choose): 
	if choose == 'inst':d_path = 'feature/inst/'
	if choose == 'pitch':d_path = 'feature/pitch/'
	sr = 16000
	f = f.replace('npz', 'mid')
	#if not (os.path.isfile(d_path+f.replace('mid', 'npz'))):
	if True:
		print(i)
		try:
			output_array = {}
			midi_data = pretty_midi.PrettyMIDI(s_path+f)
			tmp = np.load('../../database2/musescore/feature/cqt/'+f.replace('mid','npy'))
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
2. Get instrument and pitch labels
'''
def parse_midi():
	s_path = '../../database2/musescore/data/output_midi/'
	l = os.listdir(s_path)
	choose = 'pitch' #choose inst/pitch to get the label
	Parallel(n_jobs=50)(delayed(_parse_midi)(f,i,s_path,choose) for i,f in enumerate(l[:]))
#parse_midi()

def combine(y,is3D):
	y_shape = y.shape
	
	if is3D:
		num = int(y_shape[3]/2)
		output = np.zeros((y_shape[0],y_shape[1],y_shape[2],num))
	else:
		num = int(y_shape[2]/2)
		output = np.zeros((y_shape[0],y_shape[1],num))
	for n in range(num):
		if is3D:
			output[:,:,:,n] = np.sum(y[:,:,:,n*2:n*2+2], axis=3)
		else:
			output[:,:,n] = np.sum(y[:,:,n*2:n*2+2], axis=2)
		output[output>0] = 1
	return output


'''
3. Build experiment data 
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
				x = np.load('../../database2/musescore/feature/cqt/'+n.replace('mid','npy'))
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
		
		return np.vstack(x_stack),np.vstack(yi_stack),np.vstack(yp_stack),np.vstack(ys_stack)

	def create_h5(x,yi,yp,ys,name):
		hf_i = h5py.File('./ex_data/'+name+'.h5', 'a')
	
		ciy = combine(yi,False)
		cip = combine(yp,False)
		cis = combine(ys,True)
		
		hf_i.create_dataset('x', data=x, maxshape=(None, x.shape[1], x.shape[2]), chunks=True)
		hf_i.create_dataset('yi', data=yi, maxshape=(None, yi.shape[1], yi.shape[2]), chunks=True)
		hf_i.create_dataset('yp', data=yp, maxshape=(None, yp.shape[1], yp.shape[2]), chunks=True)
		hf_i.create_dataset('ys', data=ys, maxshape=(None, ys.shape[1], ys.shape[2], ys.shape[3]), chunks=True)
		print (x.shape)
		print (yi.shape)
		print (yp.shape)
		print (ys.shape)	
		

	choose = 'tr'
	file_list = "" # choose your data for file list
	x, yi, yp, ys = _get_ex(file_list,choose)
	create_h5(x, yi, yp, ys, choose)
#get_ex()
