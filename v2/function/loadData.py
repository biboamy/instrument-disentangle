#!/usr/bin/python
# -*- coding: UTF-8 -*-
import SharedArray as sa
import h5py

def load(data_name):
    
    try:
        Xtr = sa.attach('shm://%s_Xtr'%(data_name))
        Ytr = sa.attach('shm://%s_Ytr'%(data_name))
        Ytr_p = sa.attach('shm://%s_Ytr_pitch'%(data_name))
        Ytr_s = sa.attach('shm://%s_Ytr_stream'%(data_name))
    except:
        # load cqt 
        trdata = h5py.File('../ex_data/tr.h5', 'r')
        Xtr = sa.create('shm://%s_Xtr'%(data_name), (trdata['x'].shape), dtype='float32')
        Xtr[:] = trdata['x'][:]
        #load instrument label
        Ytr = sa.create('shm://%s_Ytr'%(data_name), (trdata['yi'].shape), dtype='float32')
        Ytr[:] = trdata['yi'][:]
        #load pitch label
        Ytr_p = sa.create('shm://%s_Ytr_pitch'%(data_name), (trdata['yp'].shape), dtype='float32')
        Ytr_p[:] = trdata['yp'][:]
        #load pianoroll label
        Ytr_s = sa.create('shm://%s_Ytr_stream'%(data_name), (trdata['ys'].shape), dtype='float32')
        Ytr_s[:] = trdata['ys'][:]
   
    return Xtr, Ytr, Ytr_p, Ytr_s
