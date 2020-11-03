# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:11:16 2020

Use this script to perform transfer learning. Requires a previously trained model and
labelled c3d files to add to training set.

@author: aclouthi
"""

import os
import glob
from datetime import date
import time
import pickle
import numpy as np
import torch

import functions.import_functions as iof 
import functions.nn_functions as nnf

# ---- INPUT PARAMETERS ---- #

num_epochs = 10 # Number of epochs to train for
# Path for save location of trained model
fld = os.path.join('.','data')
# Path to folder containing labelled .c3d files to add to training set
datapath = os.path.join('.','data','transfer_learning_data')
# Path to a .ckpt file of a previously trained neural network to use as base for transfer learning
modelpath = os.path.join('.','data','model_2020-10-27.ckpt')
# Path to training values from previously trained algorithm. 
# Should match the model used in modelpath.
trainvalpath = os.path.join('.','data','trainingvals_2020-10-27.pickle')
# Path to .xml file of OpenSim marker set
markersetpath = os.path.join('.','data','MarkerSet.xml')
windowSize = 120 # size of windows used to segment data for algorithm
# Markers to use to align person such that they face +x. Suggest acromions or pelvis markers.
alignMkR = 'RAC'
alignMkL = 'LAC'

# -------------------------- #
# -------------------------- #

t0 = time.time()

# Read marker set
markers, segment, uniqueSegs, segID, _, num_mks = iof.import_markerSet(markersetpath)

# Load c3d files
filelist = glob.glob(os.path.join(datapath,'*.c3d'))
data_segs, windowIdx = iof.import_labelled_c3ds(filelist,markers,alignMkR,alignMkL,windowSize)

# Load scale values and intra-segment distances
with open(trainvalpath,'rb') as f:
    trainingvals = pickle.load(f)
segdists = trainingvals['segdists']
scaleVals = trainingvals['scaleVals']
max_len = trainingvals['max_len']

# Perform transfer learning
net, running_loss = nnf.train_nn(data_segs,num_mks,max_len,windowIdx,scaleVals,
                                    num_epochs,modelpath)
  
        
with open(os.path.join(fld,'training_stats_plus' + str(len(filelist)) + 'trials_' + 
                       date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
    pickle.dump(running_loss,f)
torch.save(net.state_dict(),os.path.join(fld,'model_plus' + str(len(filelist)) + 'trials_' +
                                          date.today().strftime("%Y-%m-%d") + '.ckpt')) 


# Update intra-segment distances by adding in new training data
nframes = 0
for i in range(len(data_segs)):
    nframes = nframes + data_segs[i].shape[0]
for bs in range(len(uniqueSegs)):
    I = np.where(segID == bs)[0]
    dists = np.zeros((I.shape[0],I.shape[0],nframes))
    k = 0
    for i in range(len(data_segs)):
        pts = data_segs[i]
        for m1 in range(len(I)):
            for m2 in range(m1+1,len(I)):
                dists[m1,m2,k:k+data_segs[i].shape[0]] = (pts[:,I[m1],:] - pts[:,I[m2],:]).norm(dim=1).numpy()
        k = k+data_segs[i].shape[0]
    # update mean and std based on new data
    mn = (segdists['mean'][bs]*segdists['nframes'] + np.nanmean(dists,axis=2)*nframes) / (segdists['nframes'] + nframes) # new mean
    sumdiff = np.nansum((dists - np.repeat(np.expand_dims(segdists['mean'][bs],axis=2),nframes,axis=2))**2,axis=2)
    segdists['std'][bs] = np.sqrt(((segdists['std'][bs]**2)*(segdists['nframes']-1) + sumdiff - \
                          (segdists['nframes']+nframes)*(mn-segdists['mean'][bs])**2)/(segdists['nframes']+nframes-1))
    segdists['mean'][bs] = mn.copy()
    for i in range(1,segdists['mean'][bs].shape[0]):
        for j in range(0,i):
            segdists['mean'][bs][i,j] = segdists['mean'][bs][j,i]
            segdists['std'][bs][i,j] = segdists['std'][bs][j,i]
segdists['nframes'] = segdists['nframes']+nframes

training_vals = {'segdists' : segdists, 'scaleVals' : scaleVals}
with open(os.path.join(fld,'trainingvals_plus' + str(len(filelist)) + 'trials_' + 
                       date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
    pickle.dump(training_vals,f)
    

print('Added %d trials in %f s' % (len(filelist),time.time() - t0))
