# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 09:21:06 2020

trainAlgorithm.py

Use this script to train the marker labelling algorithm on existing labelled c3d files
or simulated marker trajectories created using generateSimTrajectories.py.

Please cite: 
    
@author: aclouthi@uottawa.ca
"""

import os
import glob
import time
from datetime import date
import pickle
from scipy import signal
import torch
import numpy as np
import pandas as pd

import functions.import_functions as fio 
import functions.nn_functions as nnf

# --------------------------------------------------------------------------- #
# ----------------------------- PARAMETERS ---------------------------------- #
# --------------------------------------------------------------------------- #


# Folder where trained model should be saved
fld = os.path.join('.','data')
# Full path to .pickle file containing simualted trajetory training data or folder
# containing labelled .c3d files to use as training data
datapath = os.path.join(fld,'simulatedTrajectories.pickle')
# Path to .xml file of OpenSim marker set
markersetpath = os.path.join(fld,'MarkerSet.xml')

fs = 240 # Sampling frequency of training data
num_epochs = 10 # Number of epochs to train for
# Markers to use to align person such that they face +x. Suggest acromions or pelvis markers.
alignMkR = 'RAC'
alignMkL = 'LAC'
# Path to a .ckpt file of a previously trained neural network if using transfer learning.
# Set to None if not using a previous model.
prevModel = None 

# --------------------------------------------------------------------------- #


t0 = time.time()

# Read marker set
markers, segment, uniqueSegs, segID, _, num_mks = fio.import_markerSet(markersetpath)

if '.pickle' in datapath:
    # Import simulated trajectory data
    with open(datapath,'rb') as f:
        data_segs = pickle.load(f)
        
    # Filter trajectories
    b, a = signal.butter(2,6,btype='low',fs=fs) # 2nd order, low-pass at 6 Hz 
    for i in range(len(data_segs)):
        for k in range(3):
            inan = torch.isnan(data_segs[i][:,:,k])
            df = pd.DataFrame(data_segs[i][:,:,k].numpy())
            df = df.interpolate(axis=0,limit_direction='both')
            dummy = torch.from_numpy(signal.filtfilt(b,a,df.to_numpy(),axis=0).copy())
            dummy[inan] = np.nan
            data_segs[i][:,:,k] = dummy 
    
    # Set windows (simulated data is already windowed)
    windowIdx = []
    for i in range(len(data_segs)):
        for m in range(num_mks):
            windowIdx.append([i,m,0,data_segs[i].shape[0]])
   
    print('Loaded simulated trajectory training data')
else:
    # Load labelled c3d files for training
    filelist = glob.glob(os.path.join(datapath,'*.c3d'))
    data_segs, windowIdx = fio.import_labelled_c3ds(filelist)
    
    print('Loaded c3ds files for training data')

# Calculate values to use to scale neural network inputs and distances between
# markers on same body segment to use for label verification
scaleVals, segdists = nnf.get_trainingVals(data_segs,uniqueSegs,segID)    

max_len = max([len(x) for x in data_segs])
training_vals = {'segdists' : segdists, 'scaleVals' : scaleVals,'max_len' : max_len}
with open(os.path.join(fld,'trainingvals_' + date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
    pickle.dump(training_vals,f)

net, running_loss = nnf.train_nn(data_segs,num_mks,max_len,windowIdx,
                                    scaleVals,num_epochs,prevModel)
    
with open(os.path.join(fld,'training_stats_' + date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
    pickle.dump(running_loss,f)
torch.save(net.state_dict(),os.path.join(fld,'model_'+ date.today().strftime("%Y-%m-%d") + '.ckpt'))  
    
print('Model saved to %s' % os.path.realpath(fld))
print('Algorithm trained in %s' % (time.time() - t0))
