# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:45:12 2020

Generate simulated marker trajectories to use for training the machine learning-
based marker labelling algorithm. Trajectories are generated based on the defined 
OpenSim (https://simtk.org/projects/opensim) marker set using body kinematics
for up to 100 participants performing a series of athletic movements.

@author: aclouthi
"""

import os
import numpy as np
import torch
import h5py
import pickle
from scipy.interpolate import CubicSpline

import functions.import_functions as io 

# --------------------------------------------------------------------------- #
# ----------------------------- PARAMETERS ---------------------------------- #
# --------------------------------------------------------------------------- #

# Path to .hdf5 file containing body kinematics of training data
bodykinpath = os.path.join('.','data','bodykinematics.hdf5')
# Path to .xml file of OpenSim marker set
markersetpath = os.path.join('.','data','MarkerSet.xml')
# Path to save .pickle file of training data
outputfile = os.path.join('.','data','simulatedTrajectories.pickle')

# Markers to use to align person such that they face +x. Suggest acromions or pelvis markers.
alignMkR = 'RAC'
alignMkL = 'LAC'
fs = 240 # Sampling frequency of data to be labelled
num_participants = 50 # number of participants to include in training data, must be <=100
max_len = 240 # Max length of data segments

# --------------------------------------------------------------------------- #

# Read marker set
markers, segment, uniqueSegs, segID, mkcoordL, num_mks = io.import_markerSet(markersetpath)

hf = h5py.File(bodykinpath,'r')

# Get body segment scaling and centre of mass
com = {}
scale = {}
for seg in uniqueSegs:
    com[seg] = np.array(hf.get('/com/'+seg))
    scale[seg] = np.array(hf.get('/scale/'+seg))

sids = list(hf['kin'].keys())


# Generate simulated trajectories 
data = []
R = np.array([[1, 0, 0],[0,0,-1],[0,1,0]]) # Converts from y=up to z=up 
k = 0
for s in range(num_participants):
    for t in hf['kin'][sids[s]].keys():
        pts = np.zeros((hf['kin'][sids[s]][t]['torso'].shape[2],len(markers),3))
        for m in range(len(markers)):
            T = np.array(hf['kin'][sids[s]][t][segment[m]])
            for i in range(T.shape[2]):
                p = np.ones((4,1))
                p[:3,0] = np.transpose((np.multiply(mkcoordL[m],scale[segment[m]][s,:]) - 
                    com[segment[m]][s,:]) * 1000)
                p = np.matmul(T[:,:,i],p)
                pts[i,m,:] = np.transpose(np.matmul(R,p[:3,0]))
        cs = CubicSpline(np.arange(0,pts.shape[0],1),pts,axis=0)
        pts = cs(np.arange(0,pts.shape[0],120/fs)) # match sampling frequency of data to label
        pts = io.align(pts,markers.index(alignMkR),markers.index(alignMkL))
        if pts.shape[0] > max_len:
            data.append(torch.from_numpy(pts[:round(pts.shape[0]/2),:,:]))
            data.append(torch.from_numpy(pts[round(pts.shape[0]/2):,:,:]))
        else:
            data.append(torch.from_numpy(pts))
    if (s+1) % 10 == 0:
        print('%d/%d complete' % (s+1,num_participants))
hf.close()

with open(outputfile,'wb') as f:
    pickle.dump(data,f)

print('Training data saved to ' + outputfile)