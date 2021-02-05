# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:49:57 2020

Algorithm for automatic labelling of motion capture markers.
Includes functions for importing motion capture data, generating simulated data, 
training the algorithm, and automatic labelling.

@author: aclouthi
"""


import xml.dom.minidom
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from scipy import signal
from scipy import stats
from scipy.optimize import linear_sum_assignment  
from scipy.interpolate import CubicSpline
from sklearn.utils.extmath import weighted_mode
from ezc3d import c3d
import h5py
import random
import copy
import pickle
import warnings
import glob
import time
from datetime import date
import os


# Import parameters
filtfreq = 6 # Cut off frequency for low-pass filter for marker trajectories

# Neural network architecture parameters
batch_size = 100
nLSTMcells = 256
nLSTMlayers = 3
LSTMdropout = .17
FCnodes = 128
# Learning parameters
lr = 0.078
momentum = 0.65


# --------------------------------------------------------------------------- #
# ---------------------------- MAIN FUNCTIONS ------------------------------- #
# --------------------------------------------------------------------------- #

def generateSimTrajectories(bodykinpath,markersetpath,outputfile,alignMkR,alignMkL,
                        fs,num_participants=100,max_len=240):
    '''
    Generate simulated marker trajectories to use for training the machine learning-
    based marker labelling algorithm. Trajectories are generated based on the defined 
    OpenSim (https://simtk.org/projects/opensim) marker set using body kinematics
    for up to 100 participants performing a series of athletic movements.

    Parameters
    ----------
    bodykinpath : string
        Path to .hdf5 file containing body kinematics of training data
    markersetpath : string
        Path to .xml file of OpenSim marker set
    outputfile : string
        Path to save .pickle file of training data
    alignMkR : string
        Markers to use to align person such that they face +x. This is for the right side.
        Suggest acromions or pelvis markers.
    alignMkL : string
        Markers to use to align person such that they face +x. This is for the left side.
        Suggest acromions or pelvis markers.
    fs : int
        Sampling frequency of data to be labelled.
    num_participants : int, optional
        Number of participants to include in training data, must be <=100. 
        The default is 100.
    max_len : int, optional
        Max length of data segments. The default is 240.

    Returns
    -------
    data : list of numpy arrays
        num_frames x num_markers x 3 matrices of marker trajectories of simulated
        training data.

    '''
    # Read marker set
    markers, segment, uniqueSegs, segID, mkcoordL, num_mks = import_markerSet(markersetpath)
    
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
            pts = align(pts,markers.index(alignMkR),markers.index(alignMkL))
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
    
    return data

def trainAlgorithm(savepath,datapath,markersetpath,fs,num_epochs=10,prevModel=None,windowSize=120,
                   alignMkR=None,alignMkL=None):
    '''
    Use this function to train the marker labelling algorithm on existing labelled 
    c3d files or simulated marker trajectories created using 
    generateSimTrajectories()

    Parameters
    ----------
    savepath : string
        Folder where trained model should be saved.
    datapath : string
        Full path to .pickle file containing simualted trajetory training data 
        or folder containing labelled .c3d files to use as training data.
    markersetpath : string
        Path to .xml file of OpenSim marker set.
    fs : int
        Sampling frequency of training data.
    num_epochs : int, optional
        Number of epochs to train for. The default is 10.
    prevModel : string, optional
        Path to a .ckpt file of a previously trained neural network if using 
        transfer learning. Set to None if not using a previous model. 
        The default is None.
    windowSize : int, optional
        Desired size of data windows. Not required if using simulated trajectories
        to train. The default is 120.
    alignMkR : string
        Markers to use to align person such that they face +x. This is for the right side.
        Suggest acromions or pelvis markers. Not required if using simulated trajectories
        to train. The default is None.
    alignMkL : string
        Markers to use to align person such that they face +x. This is for the left side.
        Suggest acromions or pelvis markers. Not required if using simulated trajectories
        to train. The default is None.

    Returns
    -------
    None.

    '''
    t0 = time.time()
 
    # Read marker set
    markers, segment, uniqueSegs, segID, _, num_mks = import_markerSet(markersetpath)
    
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
                
        max_len = max([len(x) for x in data_segs])
       
        print('Loaded simulated trajectory training data')
    else:
        # Load labelled c3d files for training
        filelist = glob.glob(os.path.join(datapath,'*.c3d'))
        data_segs, windowIdx = import_labelled_c3ds(filelist,markers,
                                                        alignMkR,alignMkL,windowSize)
        
        max_len = max([x[3]-x[2] for x in windowIdx])
        
        print('Loaded c3ds files for training data')
    
    # Calculate values to use to scale neural network inputs and distances between
    # markers on same body segment to use for label verification
    scaleVals, segdists = get_trainingVals(data_segs,uniqueSegs,segID)    
    
    max_len = max([len(x) for x in data_segs])
    training_vals = {'segdists' : segdists, 'scaleVals' : scaleVals,'max_len' : max_len}
    with open(os.path.join(savepath,'trainingvals_' + date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
        pickle.dump(training_vals,f)
    
    net, running_loss = train_nn(data_segs,num_mks,max_len,windowIdx,
                                        scaleVals,num_epochs,prevModel)
        
    with open(os.path.join(savepath,'training_stats_' + date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
        pickle.dump(running_loss,f)
    torch.save(net.state_dict(),os.path.join(savepath,'model_'+ date.today().strftime("%Y-%m-%d") + '.ckpt'))  
        
    print('Model saved to %s' % os.path.realpath(savepath))
    print('Algorithm trained in %s' % (time.time() - t0))


def transferLearning(savepath,datapath,modelpath,trainvalpath,markersetpath,
                     num_epochs=10,windowSize=120,alignMkR=None,alignMkL=None):
    '''
    Use this function to perform transfer learning. Requires a previously trained 
    model and labelled c3d files to add to training set.

    Parameters
    ----------
    savepath : string
        Path for save location of trained model.
    datapath : string
        Path to folder containing labelled .c3d files to add to training set.
    modelpath : string
        Path to a .ckpt file of a previously trained neural network to use as 
        base for transfer learning.
    trainvalpath : string
        Path to training values from previously trained algorithm. 
        Should match the model used in modelpath.
    markersetpath : string
        Path to .xml file of OpenSim marker set.
    num_epochs : int, optional
        Number of epochs to train for. The default is 10.
    windowSize : int, optional
        Size of windows used to segment data for algorithm. The default is 120.
    alignMkR : string, optional
        Markers to use to align person such that they face +x. This is for the right side.
        Suggest acromions or pelvis markers. The default is None (ie. no alignment).
    alignMkL : string, optional
        Markers to use to align person such that they face +x. This is for the left side.
        Suggest acromions or pelvis markers. The default is None (ie. no alignment).

    Returns
    -------
    None.

    '''
    
    t0 = time.time()

    # Read marker set
    markers, segment, uniqueSegs, segID, _, num_mks = import_markerSet(markersetpath)
    
    # Load c3d files
    filelist = glob.glob(os.path.join(datapath,'*.c3d'))
    data_segs, windowIdx = import_labelled_c3ds(filelist,markers,alignMkR,alignMkL,windowSize)
    
    # Load scale values and intra-segment distances
    with open(trainvalpath,'rb') as f:
        trainingvals = pickle.load(f)
    segdists = trainingvals['segdists']
    scaleVals = trainingvals['scaleVals']
    max_len = trainingvals['max_len']
    
    # Perform transfer learning
    net, running_loss = train_nn(data_segs,num_mks,max_len,windowIdx,scaleVals,
                                        num_epochs,modelpath)
      
            
    with open(os.path.join(savepath,'training_stats_plus' + str(len(filelist)) + 'trials_' + 
                           date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
        pickle.dump(running_loss,f)
    torch.save(net.state_dict(),os.path.join(savepath,'model_plus' + str(len(filelist)) + 'trials_' +
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
    
    training_vals = {'segdists' : segdists, 'scaleVals' : scaleVals,'max_len' : max_len}
    with open(os.path.join(savepath,'trainingvals_plus' + str(len(filelist)) + 'trials_' + 
                           date.today().strftime("%Y-%m-%d") + '.pickle'),'wb') as f:
        pickle.dump(training_vals,f)
        
    
    print('Added %d trials in %f s' % (len(filelist),time.time() - t0))


# --------------------------------------------------------------------------- #
# --------------------------- IMPORT FUNCTIONS ------------------------------ #
# --------------------------------------------------------------------------- #

def import_markerSet(markersetpath):
    '''
    Read marker set from OpenSim marker set .xml file

    Parameters
    ----------
    markersetpath : string
        path to .xml file

    Returns
    -------
    markers : list of strings
        marker names
    segment : list of strings
        body segment each marker belongs to
    uniqueSegs : list of strings
        body segments
    segID : list of ints
        index of body segment each marker belongs to
    mkcoordL : list of numpy arrays
        position of each marker relative to the local coordinate system of its 
        body segment
    num_mks : int
        number of markers

    '''
    markersetxml = xml.dom.minidom.parse(markersetpath);
    mkxml=markersetxml.getElementsByTagName('Marker')
    markers = []
    segment = []
    mkcoordL = []
    for i in range(len(mkxml)):
        markers.append(mkxml[i].attributes['name'].value)
        segment.append(mkxml[i].childNodes[3].childNodes[0].data)
        mkcoordL.append(np.fromstring(mkxml[i].childNodes[7].childNodes[0].data,sep=' '))
    segment = [x.split('/')[-1] for x in segment]
    uniqueSegs = sorted(list(set(segment)))
    
    segID = -1 * np.ones(len(segment),dtype=np.int64)
    for i in range(len(segment)):
        segID[i] = uniqueSegs.index(segment[i])
    num_mks = len(markers)
    
    return markers, segment, uniqueSegs, segID, mkcoordL, num_mks
    
def align(data,m1,m2):
    '''
    Rotate points about z-axis (vertical) so that participant is facing +x direction.
    Angle to rotate is calculated based on m1 and m2. These should be the indices of 
    a marker on the right and left side of the torso or head.
    If one of these markers is missing from the entire trial, the data will not be 
    rotated.

    Parameters
    ----------
    data : numpy array
        num_frames x num_markers x 3 matrix of marker trajectories
    m1 : int
        index of the right side marker
    m2 : int
        index of the left side marker

    Returns
    -------
    data : numpy array
        Rotated marker trajectories
    '''
    # if alignment markers are missing for entire trial, can't align
    if np.isnan(data[:,m1,0]).sum() == data.shape[0] or np.isnan(data[:,m2,0]).sum() == data.shape[0]:
        return data
    else:
        # find first non-nan entry for the markers
        i = 0
        while np.isnan(data[i,m1,0]) or np.isnan(data[i,m2,0]):
            i = i+1        
        pts = data[i,:,:]
        v = pts[m2,:] - pts[m1,:] # L - R
        theta = np.arctan2(v[0],v[1])       
        T = np.array([[np.cos(theta),-np.sin(theta),0],
                      [np.sin(theta),np.cos(theta),0],
                      [0,0,1]])    
        dataR = np.empty(data.shape)
        for i in range(0,data.shape[0]):
            pts = data[i,:,:]
            ptsR = np.transpose(np.matmul(T,pts.transpose()))
            dataR[i,:,:] = ptsR
        
        return dataR

def window_data(data_segs,windowSize,num_mks):
    '''
    Determine how to window data. 

    Parameters
    ----------
    data_segs : list of numpy arrays
        num_frames x num_markers x 3 arrays of marker trajectories imported from .c3d files
    windowSize : int
        desired size of windows
    num_mks : int
        number of markers of interest
        windows will be created for the first num_mks trajectories

    Returns
    -------
    windowIdx : list of lists
        indices to use to window data, required input to training function

    '''
    
    windowIdx = []
    for t in range(len(data_segs)):
        pts = data_segs[t]
        if torch.is_tensor(pts):
            pts = pts.numpy()
        for m in range(num_mks):
            # only include if it's not all nans
            if len(np.where(~np.isnan(pts[:,m,0]))[0]) > 0: 
                i1 = np.where(~np.isnan(pts[:,m,0]))[0][0] # first visible frame
                while i1 < pts.shape[0]:
                    if (np.isnan(pts[i1:,m,0])).sum() > 0: # if any more nans
                        i2 = np.where(np.isnan(pts[i1:,m,0]))[0][0] + i1
                    else:
                        i2 = pts.shape[0]
                    while i1 <= i2:
                        if (i2 - (i1+windowSize) < 12) or (i1 + windowSize > i2):
                            if i2 - i1 > 0:
                                # confirm that there are other markers visible in this window
                                if (~np.isnan(np.concatenate((pts[i1:i2,0:m,:],pts[i1:i2,m+1:,:]),1))).sum() > 0:
                                    windowIdx.append([t,m,i1,i2])
                            if  (~np.isnan(pts[i2:,m,0])).sum() > 1: # any more visible markers?
                                i1 = i2 + np.where(~np.isnan(pts[i2:,m,0]))[0][0]
                            else: 
                                i1 = pts.shape[0] + 1
                        else:
                            if (~np.isnan(np.concatenate((pts[i1:i2,0:m,:],pts[i1:i2,m+1:,:]),1))).sum() > 0:
                                windowIdx.append([t,m,i1,i1+windowSize])
                            i1 = i1 + windowSize
    return windowIdx



def import_labelled_c3ds(filelist,markers,alignMkR,alignMkL,windowSize):
    '''
    Import c3d files for training, sort markers to match marker set order, 
    filter data, rotate data such that person faces +x, and determine window 
    indices.
    

    Parameters
    ----------
    filelist : list of strings
        list of filepaths to .c3d files to import
    markers : list of strings
        list of marker names
    alignMkR : string
        name of marker to use to rotate data on RIGHT side of body,
        set to None if rotation is not needed
    alignMkL : string
        name of marker to use to rotate data on LEFT side of body,
        set to None if rotation is not needed
    windowSize : int
        desired size of windows

    Returns
    -------
    data_segs : list of torch tensors
        num_frames x num_markers x 3 tensors of marker trajectories imported from .c3d files
    windowIdx : list of lists
        indices to use to window data, required input to training function

    '''
    num_mks = len(markers)
    data_segs = []
    for trial in filelist:
        # Import c3d and reorder points according to marker set order
        c3ddat = c3d(trial)
        alllabels = c3ddat['parameters']['POINT']['LABELS']['value']
        fs = c3ddat['parameters']['POINT']['RATE']['value'][0]
        pts = np.nan * np.ones((c3ddat['data']['points'].shape[2],num_mks,3))
        for i in range(c3ddat['data']['points'].shape[1]):
            j = [ii for ii,x in enumerate(markers) if x in alllabels[i]]
            if len(j) == 0:
                # if this is an extraneous marker (not part of the defined marker set),
                # add to the end of the array
                dummy = np.empty((pts.shape[0],1,3))
                for k in range(3):
                    dummy[:,0,k] = c3ddat['data']['points'][k,i,:]
                pts = np.append(pts,dummy,axis=1)
            elif len(j) == 1:
                # If this is part of the marker set
                for k in range(3):
                    pts[:,j[0],k] = c3ddat['data']['points'][k,i,:]
        
        # delete any empty frames at the end
        while (~np.isnan(pts[-1,:,0])).sum() == 0:
            pts = pts[:-1,:,:]
            
        # rotate so that the person faces +x
        if (alignMkR is not None) and (alignMkL !='') and (alignMkL is not None) and (alignMkL !=''):
            pts = align(pts,markers.index(alignMkR),markers.index(alignMkL))
            
        # Filter with 2nd order, low-pass Butterworth at filtfreq Hz 
        b, a = signal.butter(2,filtfreq,btype='low',fs=fs) 
        for k in range(3):
            inan = np.isnan(pts[:,:,k])
            df = pd.DataFrame(pts[:,:,k])
            df = df.interpolate(axis=0,limit_direction='both')
            dummy = signal.filtfilt(b,a,df.to_numpy(),axis=0).copy()
            dummy[inan] = np.nan
            pts[:,:,k] = dummy
        data_segs.append(torch.from_numpy(pts))
    
    windowIdx = window_data(data_segs,windowSize,num_mks)
        
    return data_segs, windowIdx

def import_raw_c3d(file,rotang):
    '''
    Import an a c3d file for labelling. Trajectories are split at gaps were the
    distance the marker travels during the occlusion is greater than the distance to 
    the closest marker in the next frame. Marker data is rotated 'rotang' degrees
    about the z-axis (vertical).

    Parameters
    ----------
    file : string
        path to the c3d file to be imported
    rotang : float
        Angle to rotate the marker data about z-axis in DEGREES. 

    Returns
    -------
    pts : numpy array
        num_frames x num_markers x 3 array of marker trajectories imported from .c3d files
    fs : float
        sampling frequency used in c3d file

    '''
    
    c3ddat = c3d(file) # read in c3d file
    rawpts = c3ddat['data']['points'][0:3,:,:].transpose((2,1,0)) # Get points from c3d file
    fs = c3ddat['parameters']['POINT']['RATE']['value'] # sampling frequency
    rawlabels = c3ddat['parameters']['POINT']['LABELS']['value']
    
    # # Try to find and fix places where the markers swap indices
    # thresh = 20
    # for m in range(rawpts.shape[1]):
    #     kf = np.where(np.isnan(rawpts[1:,m,0]) != np.isnan(rawpts[0:-1,m,0]))[0]
    #     if ~np.isnan(rawpts[0,m,0]):
    #         kf = np.insert(kf,0,-1,axis=0)
    #     if ~np.isnan(rawpts[-1,m,0]):
    #         kf = np.concatenate((kf,[rawpts.shape[0]-1]))
    #     kf = np.reshape(kf,(-1,2))
    #     k = 0
    #     while k < kf.shape[0]-1:
    #         d = np.linalg.norm(rawpts[kf[k+1,0]+1,m,:] - rawpts[kf[k,1],m,:])
    #         all_d = np.linalg.norm(rawpts[kf[k,1]+1,:,:] - rawpts[kf[k,1],m,:],axis=1)
    #         all_d[m] = np.nan
    #         if (~np.isnan(all_d)).sum() > 0:
    #             if d > np.nanmin(all_d) and np.nanmin(all_d) < thresh and \
    #                     np.isnan(rawpts[kf[k,1],np.nanargmin(all_d),0]):
    #                 dummy = rawpts[kf[k,1]+1:,m,:].copy()
    #                 rawpts[kf[k,1]+1:,m,:] = rawpts[kf[k,1]+1:,np.nanargmin(all_d),:]
    #                 rawpts[kf[k,1]+1:,np.nanargmin(all_d),:] = dummy.copy()
                    
    #                 kf = np.where(np.isnan(rawpts[1:,m,0]) != np.isnan(rawpts[0:-1,m,0]))[0]
    #                 if ~np.isnan(rawpts[0,m,0]):
    #                     kf = np.insert(kf,0,0,axis=0)
    #                 if ~np.isnan(rawpts[-1,m,0]):
    #                     kf = np.concatenate((kf,[rawpts.shape[0]-1]))
    #                 kf = np.reshape(kf,(-1,2))
    #         k = k+1
            
    # Wherever there is a gap, check if the marker jumps further than the distance to the 
    # next closest marker. If so, split it into a new trajectory.
    pts = np.empty((rawpts.shape[0],0,3))
    labels = []
    for m in range(rawpts.shape[1]):    
        # key frames where the marker appears or disappears  
        kf = np.where(np.isnan(rawpts[1:,m,0]) != np.isnan(rawpts[0:-1,m,0]))[0]
        if ~np.isnan(rawpts[0,m,0]):
            kf = np.insert(kf,0,-1,axis=0)
        if ~np.isnan(rawpts[-1,m,0]):
            kf = np.concatenate((kf,[rawpts.shape[0]-1]))
        kf = np.reshape(kf,(-1,2))
        k = 0
        while k < kf.shape[0]:
            i1 = kf[k,0]
            d = 0
            gapsize = 0
            min_d = 1000
            while d < min_d and gapsize < 60:
                if k < kf.shape[0]-1:
                    d = np.linalg.norm(rawpts[kf[k+1,0]+1,m,:] - rawpts[kf[k,1],m,:])
                    all_d = np.linalg.norm(rawpts[kf[k,1]+1,:,:] - rawpts[kf[k,1],m,:],axis=1)
                    all_d[m] = np.nan
                    if (~np.isnan(all_d)).sum() > 0:
                        min_d = np.nanmin(all_d)
                    else:
                        min_d = 1000
                    gapsize = kf[k+1,0] - kf[k,1]
                else:
                    gapsize = 61
                k=k+1
            if kf[k-1,1] - i1 > 2:
                traj = np.nan * np.ones((rawpts.shape[0],1,3))
                traj[i1+1:kf[k-1,1]+1,0,:] = rawpts[i1+1:kf[k-1,1]+1,m,:] 
                pts = np.append(pts,traj,axis=1)
                labels.append(rawlabels[m])
                

    # Angle to rotate points about z-axis
    rotang = float(rotang) * np.pi/180 
    Ralign = np.array([[np.cos(rotang),-np.sin(rotang),0],
                       [np.sin(rotang),np.cos(rotang),0],
                       [0,0,1]])
    for i in range(pts.shape[1]):
        pts[:,i,:] = np.matmul(Ralign,pts[:,i,:].transpose()).transpose()
                
    return pts, fs, labels


# --------------------------------------------------------------------------- #
# ------------------------- NEURAL NET FUNCTIONS ---------------------------- #
# --------------------------------------------------------------------------- #

def get_trainingVals(data_segs,uniqueSegs,segID):
    '''
    Calculates the values that will be used to scale the input matrix to the neural
    network. These are the mean observed relative distances, velocities, and 
    accelerations from 2000 trials from the training set.
    
    Calculates the mean and standard deviation of distances among markers belonging
    to each body segment. These are used to validate and correct the labels 
    predicted by the neural network.

    Parameters
    ----------
    data_segs : list of torch tensors
        num_frames x num_markers x 3 tensors of marker trajectories in training set
    uniqueSegs : list of strings
        body segment names
    segID : list of ints
        index of body segments each marker belongs to

    Returns
    -------
    scaleVals : list of floats
       mean relative distance, velocity, and acceleration in training set and
       number of data frames used to calculate these
    segdists : dictionary
        ['mean'] : numpy arrays for each body segment containing mean distances
                   among associated markers in training set
        ['std'] : numpy arrays for each body segment containing standard deviation
                   of distances among associated markers in training set
        ['nframes'] : number of frames used to calculate these values

    '''
    # Get scale values
    sumDist = 0.0
    sumVel = 0.0
    sumAccn = 0.0
    nDist = 0.0
    nVel = 0.0
    nAccn = 0.0

    # Only use 2000 segments to save computing time
    if len(data_segs) > 2000:
        I = random.sample(range(len(data_segs)),2000)
    else:
        I = range(len(data_segs))
    for i in I:
        for m in range(int(data_segs[i].shape[1])):
            # marker distances relative to marker m
            xyz = data_segs[i] - data_segs[i][:,m,:].unsqueeze(1).repeat(1,data_segs[i].shape[1],1)
            xyz_v = xyz[1:,:,:] - xyz[0:xyz.shape[0]-1,:,:]
            xyz_v_norm = xyz_v.norm(dim=2)
            xyz_a = xyz_v[1:,:,:] - xyz_v[0:xyz_v.shape[0]-1,:,:]
            xyz_a_norm = xyz_a.norm(dim=2)
            
            sumDist = sumDist + np.nansum(abs(xyz))
            nDist = nDist + (~torch.isnan(xyz[:,0:m,:])).sum() + (~torch.isnan(xyz[:,m+1:,:])).sum() #xyz.shape[0]*(xyz.shape[1]-1)*xyz.shape[2]
            sumVel = sumVel + np.nansum(xyz_v_norm)
            nVel = nVel + (~torch.isnan(xyz_v_norm[:,0:m])).sum() + (~torch.isnan(xyz_v_norm[:,m+1:])).sum() #xyz_v_norm.shape[0] * (xyz_v_norm.shape[1]-1)
            sumAccn = sumAccn + np.nansum(xyz_a_norm)
            nAccn = nAccn + (~torch.isnan(xyz_a_norm[:,0:m])).sum() + (~torch.isnan(xyz_a_norm[:,m+1:])).sum() # xyz_a_norm.shape[0] * (xyz_a_norm.shape[1]-1)
    
    scaleVals = [sumDist/nDist, sumVel/nVel, sumAccn/nAccn,nDist,nVel,nAccn]
    
    
    # Calculate distances between markers on same body segments
    dists_mean = []
    dists_std = []
    nframes = 0
    for i in range(len(data_segs)):
        nframes = nframes + data_segs[i].shape[0]
    for bs in range(len(uniqueSegs)):
        I = np.where(segID == bs)[0] # indices of markers on this body segment
        dists = np.zeros((I.shape[0],I.shape[0],nframes))
        dists_mean.append(np.zeros((I.shape[0],I.shape[0])))
        dists_std.append(np.zeros((I.shape[0],I.shape[0])))
        k = 0
        for i in range(len(data_segs)):
            pts = data_segs[i]
            for m1 in range(len(I)):
                for m2 in range(m1+1,len(I)):
                    dists[m1,m2,k:k+data_segs[i].shape[0]] = \
                        (pts[:,I[m1],:] - pts[:,I[m2],:]).norm(dim=1).numpy()
            k = k+data_segs[i].shape[0]
        dists_mean[bs] = dists.mean(axis=2)
        dists_std[bs] = dists.std(axis=2)
        for i in range(1,dists_mean[bs].shape[0]):
            for j in range(0,i):
                dists_mean[bs][i,j] = dists_mean[bs][j,i]
                dists_std[bs][i,j] = dists_std[bs][j,i]
    segdists = {'mean' : dists_mean,'std' : dists_std,'nframes' : nframes}
        
    return scaleVals, segdists

# Generates the input data for the neural network.
class markerdata(torch.utils.data.Dataset):
    def __init__(self,marker_data,num_mks,windowIdx,scaleVals):
        self.marker_data = copy.deepcopy(marker_data)
        self.num_mks = num_mks # Number of marker labels
        self.windowIdx = windowIdx
        self.scaleVals = scaleVals
    def __len__(self): # Should be the number of items in this dataset
        return len(self.windowIdx)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() # index will be row major 
        num_mks = self.num_mks 
        t = self.windowIdx[idx][0]
        m = self.windowIdx[idx][1]
        i1 = self.windowIdx[idx][2]
        i2 = self.windowIdx[idx][3]
        
        xyz_raw = copy.deepcopy(self.marker_data[t][i1:i2,:,:])
        xyz_m = xyz_raw[:,m,:] # current marker    
        xyz_raw = torch.cat((xyz_raw[:,0:m,:],xyz_raw[:,m+1:,:]),dim=1)    
        # check what is visible at this time and take the markers that are 
        # visible for the greatest number of frames
        mksvis = (~torch.isnan(xyz_raw[:,:,0])).sum(dim=0)
        if (mksvis == xyz_raw.shape[0]).sum() > num_mks:
            # then also sort by distance
            xyz_raw = xyz_raw[:,mksvis==xyz_raw.shape[0],:]
            d = (xyz_raw - xyz_m.unsqueeze(1).repeat(1,xyz_raw.shape[1],1)).norm(dim=2)
            _,I = (d.mean(0)).sort()
            xyz_raw = xyz_raw[:,I[0:num_mks-1],:]
        else:
            _,I = mksvis.sort(descending=True)
            xyz_raw = xyz_raw[:,I[0:num_mks-1],:]
        # Fill in any missing markers with the mean of all of the other visible markers
        if torch.isnan(xyz_raw[:,:,0]).sum() > 0:
            inan = torch.where(torch.isnan(xyz_raw))
            xyz_raw[inan] = torch.take(torch.from_numpy(np.nanmean(xyz_raw,1)),
                                       inan[0]*3+inan[2])
            if torch.isnan(xyz_raw[:,:,0]).any(1).sum() > 0:
                # if there somehow ended up to be empty frames, delete them
                xyz_m = xyz_m[~torch.isnan(xyz_raw[:,:,0]).any(1),:]
                xyz_raw = xyz_raw[~torch.isnan(xyz_raw[:,:,0]).any(1),:,:]
        xyz_raw = xyz_raw - xyz_m.unsqueeze(1).repeat(1,xyz_raw.shape[1],1)
        d = (xyz_raw.mean(dim=0)).norm(dim=1)
        _, I = d.sort() # Sort trajectories by distance relative to marker m
        xyz = xyz_raw[:,I,:]
   
        # Add in velocity and accn
        xyz_v = torch.zeros(xyz.shape,dtype=xyz.dtype)
        xyz_v[1:,:,:] = xyz[1:,:,:] - xyz[0:xyz.shape[0]-1,:,:]
        xyz_v_norm = xyz_v.norm(dim=2)
        if xyz_v_norm.shape[0] > 1:
            xyz_v_norm[0,:] = xyz_v_norm[1,:]
        xyz_a = torch.zeros(xyz.shape,dtype=xyz.dtype)
        xyz_a[1:,:,:] = xyz_v[1:xyz_v.shape[0],:,:] - xyz_v[0:xyz_v.shape[0]-1,:,:]
        xyz_a_norm = xyz_a.norm(dim=2)
        if xyz_a_norm.shape[0] > 2:
            xyz_a_norm[1,:] = xyz_a_norm[2,:]
            xyz_a_norm[0,:] = xyz_a_norm[2,:]
        
        # Scale input data
        xyz = xyz / self.scaleVals[0]
        xyz_v_norm = xyz_v_norm / self.scaleVals[1]
        xyz_a_norm = xyz_a_norm / self.scaleVals[2]
        out = torch.cat((xyz,xyz_v_norm.unsqueeze(2),xyz_a_norm.unsqueeze(2)),2)
        out = out.reshape(-1,(num_mks-1)*5)
        
        return out, m, idx

# Collate function for data loader. Pads the data to make it equally sized.
def pad_collate(batch):
    batch = list(filter(lambda xx:xx[0] is not None,batch))
    (X,Y,T) = zip(*batch)
    # filter out None entries
    x_lens = [len(x) for x in X]
    Y_out = [y for y in Y]
    T_out = [t for t in T]
    X_pad = nn.utils.rnn.pad_sequence(X,batch_first=True,padding_value=0)
    return X_pad, Y_out, T_out, x_lens

# Define network architecture
class Net(nn.Module):
    def __init__(self, max_len,num_mks):
        super(Net,self).__init__()
        self.max_len = max_len
        self.lstm = nn.LSTM((num_mks-1)*5,nLSTMcells,num_layers=nLSTMlayers,dropout=LSTMdropout)
        self.fc = nn.Sequential(nn.Linear(max_len*nLSTMcells,FCnodes),
                            nn.BatchNorm1d(FCnodes),
                            nn.ReLU(),
                            nn.Linear(FCnodes,num_mks))
    def forward(self,x,x_lens):
        out = torch.nn.utils.rnn.pack_padded_sequence(x.float(),x_lens,batch_first=True,enforce_sorted=False)
        out, (h_t,h_c) = self.lstm(out)
        out,_ = torch.nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=self.max_len)
        out = self.fc(out.view(out.shape[0],-1))
        return out

def train_nn(data_segs,num_mks,max_len,windowIdx,scaleVals,num_epochs,prevModel):
    '''
    Train the neural network. 
    Will use GPU if available.

    Parameters
    ----------
    data_segs : list of torch tensors
        num_frames x num_markers x 3 tensors of marker trajectories in training set
    num_mks : int
        number of markers in marker set
    max_len : int
        maximum window length
    windowIdx : list of lists
        indices to use to window data, required input to training function
    scaleVals : list of floats
       mean relative distance, velocity, and acceleration in training set and
       number of data frames used to calculate these. Used to scale variables 
       before inputting to neural network.
    num_epochs : int
        number of epoch to train for
    prevModel : string
        path to the .ckpt file for a previously trained model if using transfer 
        learning
        set to None if not using transfer learning

    Returns
    -------
    net : torch.nn.Module
        trained neural network
    running_loss: list of floats
        running loss for network training

    '''
    
    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    
    # Create dataset and torch data loader
    traindata = markerdata(data_segs,num_mks,windowIdx,scaleVals)
    trainloader = torch.utils.data.DataLoader(traindata,batch_size=batch_size,
                                              shuffle=True,collate_fn=pad_collate)
    # Create neural net
    net = Net(max_len,num_mks).to(device)
    
    # Load previous model if transfer learning
    if (prevModel is not None) and (prevModel != ''):
        net.load_state_dict(torch.load(prevModel,map_location=device)) 
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    # Train Network
    total_step = len(trainloader)
    running_loss = []
    for epoch in range(num_epochs):
        for i, (data, labels, trials, data_lens) in enumerate(trainloader):
            data = data.to(device)
            labels = torch.LongTensor(labels)
            labels = labels.to(device)
            
            # Forward pass
            outputs = net(data,data_lens)
            loss = criterion(outputs,labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss.append(loss.item())
            
            # Print stats
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1,num_epochs,i+1,
                                                                         total_step,loss.item()))
    return net, running_loss

def predict_nn(modelpath,pts,windowIdx,scaleVals,num_mks,max_len):
    '''
    Run the neural network to get label probabilities

    Parameters
    ----------
    modelpath : string
        path to .ckpt file of trained neural network weights
    pts : numpy array
        num_frames x num_markers x 3 array of marker trajectories to be labelled
    windowIdx : list of lists
        indices to use to window data, required input to training function
    scaleVals : list of floats
       mean relative distance, velocity, and acceleration in training set and
       number of data frames used to calculate these. Used to scale variables 
       before inputting to neural network.
    num_mks : int
        number of markers in marker set
    max_len : int
        max length of data windows

    Returns
    -------
    probWindow : torch tensor
        num_windows x num_mks tensor of label probabilities for each window

    '''
    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load trained network weights
    net = Net(max_len,num_mks).to(device)
    net.load_state_dict(torch.load(modelpath,map_location=device)) 
    
    dataset = markerdata([torch.from_numpy(pts)],num_mks,windowIdx,scaleVals)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                             shuffle=False,collate_fn=pad_collate)
    
    # Apply neural net
    sm = nn.Softmax(dim=1)
    net.eval() 
    with torch.no_grad():
        probWindow = torch.zeros(len(windowIdx),num_mks)
        k = 0
        for data, trajNo, segIdx, data_lens in dataloader:
            if data is not None:
                data = data.to(device)
                outputs = net(data, data_lens)
                _,predicted = torch.max(outputs.data,1)
                
                # Get probabilities for each window
                outputs = sm(outputs)
                probWindow[k:k+data.shape[0],:] = outputs
                k = k + data.shape[0]
                
    return probWindow


# --------------------------------------------------------------------------- #
# ---------------------------- LABEL FUNCTIONS ------------------------------ #
# --------------------------------------------------------------------------- #
    
def marker_label(pts,modelpath,trainvalpath,markersetpath,fs,windowSize):
    '''

    Parameters
    ----------
    pts : numpy array 
        [num_frames x num_markers x 3] array of marker trajectories to be labelled
    modelpath : string
        path to .ckpt file containing trained neural network weights
    trainvalpath : string
        path to .pickle file containing the training values obtained from trainAlgorithm.py
    markersetpath : string
        path to .xml file containing OpenSim marker set definition
    fs : float
        sampling frequency of data in pts
    windowSize : int
        desired size of windows

    Returns
    -------
    labels_predicted : list of strings
        predicted labels for each marker trajectory
    confidence : numpy array
        [1 x num_trajectories] array of confidence in predicted label
    Y_pred : torch tensor
        [1 x num_trajectories] tensor of predicted label indices

    '''
    # Read marker set
    markers, segment, uniqueSegs, segID, mkcoordL, num_mks = import_markerSet(markersetpath)
    
    # Get expected inter-marker distances organized by segment
    with open(trainvalpath,'rb') as f:
        trainingvals = pickle.load(f)
    segdists = trainingvals['segdists']
    scaleVals = trainingvals['scaleVals']
    max_len = trainingvals['max_len']
    
    pts = np.array(pts,dtype=np.float64)
    
    num_mks = len(markers)
    
    # Fill small gaps
    gap_idx = np.isnan(pts)
    for m in range(pts.shape[1]):
        df = pd.DataFrame(pts[:,m,:])
        df = df.interpolate(axis=0,limit=4,limit_area='inside')
        pts[:,m,:] = df
    
    # Filter
    b, a = signal.butter(2,6,btype='low',fs=fs) # 2nd order, low-pass at 6 Hz    
    for k in range(3):
        inan = np.isnan(pts[:,:,k])
        df = pd.DataFrame(pts[:,:,k])
        df = df.interpolate(axis=0,limit_direction='both')
        dummy = signal.filtfilt(b,a,df.to_numpy(),axis=0).copy()
        dummy[inan] = np.nan
        pts[:,:,k] = dummy

    # If there are fewer trajectories than the expected number of markers, add some empty columns
    if pts.shape[1] < num_mks:
        pts = np.concatenate((pts,np.nan * np.ones((pts.shape[0],num_mks-pts.shape[1],3),
                                                    dtype=pts.dtype)),axis=1)
    
    # --- Initial Label Prediction --- #
    
    # Determine window indices
    windowIdx = window_data([pts],windowSize,pts.shape[1])
    
    # Apply neural network to get label probablities within windows
    probWindow = predict_nn(modelpath,pts,windowIdx,scaleVals,num_mks,max_len)
    
    # Convert to frame-by-frame probabilities
    probFrame = torch.zeros(pts.shape[1],num_mks,pts.shape[0])
    for t in range(len(windowIdx)):
        probFrame[windowIdx[t][1],:,windowIdx[t][2]:windowIdx[t][3]] = \
            probWindow[t,:].repeat(windowIdx[t][3]-windowIdx[t][2],1).t().unsqueeze(dim=0)

    # Find all frames where any marker appears or disappears
    keyframes = [0]
    for m in range(pts.shape[1]):
        I = np.where(np.isnan(pts[1:,m,0]) != np.isnan(pts[0:-1,m,0]))[0]
        for i in range (len(I)):
            keyframes.append(I[i])
    keyframes = sorted(set(keyframes))
    if keyframes[-1] < pts.shape[0]:
        keyframes.append(pts.shape[0])
    
    # Make some new windows based on keyframes
    # These are guaranteed to have only one of each marker in them
    prob = torch.zeros((probFrame.shape[0],probFrame.shape[1],len(keyframes)-1),
                        dtype=probFrame.dtype)
    if len(keyframes) == 1:
        prob = probFrame.mean(dim=2).unsqueeze(2)
    else:
        for i in range(len(keyframes)-1):
            prob[:,:,i] = probFrame[:,:,keyframes[i]:keyframes[i+1]].mean(dim=2) 
           
    # Hungarian Algorithm to assign labels within new windows
    y_pred = -1 * torch.ones((prob.shape[2],prob.shape[0]),dtype=torch.int64) # predicted label
    confidence = torch.zeros(prob.shape[2],prob.shape[0],dtype=prob.dtype) 
    for t in range(prob.shape[2]):
        p = copy.deepcopy(prob[:,:,t])
    
        with np.errstate(divide='ignore'):
            alpha = np.log(((1-p)/p).detach().numpy())
        alpha[alpha==np.inf] = 100
        alpha[alpha==-np.inf] = -100
        
        R,C = linear_sum_assignment(alpha) # Hungarian Algorithm
                
        for i in range(R.shape[0]):
            y_pred[t,R[i]] = int(C[i])
            confidence[t,R[i]] = prob[R[i],C[i],t]
    
    # Convert to frame-by-frame label prediction and confidence
    y_pred_frame = -1 * torch.ones((pts.shape[0],pts.shape[1]),dtype=torch.int64)
    confidence_frame = torch.empty((pts.shape[0],pts.shape[1]),dtype=prob.dtype)
    if len(keyframes) == 1:
        y_pred_frame = y_pred.repeat(pts.shape[0],1)
        for d in range(pts.shape[1]):
            confidence_frame[:,d] = probFrame[d,y_pred[0,d],:]
    else:
        for t in range(len(keyframes)-1):
            y_pred_frame[keyframes[t]:keyframes[t+1],:] = y_pred[t,:]
            for d in range(pts.shape[1]):
                confidence_frame[keyframes[t]:keyframes[t+1],d] = probFrame[d,y_pred[t,d],keyframes[t]:keyframes[t+1]]

    # Calculate scores for each trajectory using weighted mode
    Y_pred = -1 * torch.ones(pts.shape[1],dtype=y_pred_frame.dtype)
    confidence_final = np.empty(pts.shape[1])
    confidence_weight = np.empty(pts.shape[1])
    for d in range(pts.shape[1]):
        a,b = weighted_mode(y_pred_frame[:,d],confidence_frame[:,d])
        Y_pred[d] = torch.from_numpy(a)
        confidence_final[d] = confidence_frame[y_pred_frame[:,d]==a[0],d].mean()
        confidence_weight[d] = b
    
    # Replace original gaps so that this doesn't interfere with error checking
    pts[gap_idx] = np.nan
    
    # --- Error checking and correction --- #
    
    # Remove labels where inter-marker distances within the segment aren't within expected range
    for bs in range(len(uniqueSegs)):
        I = np.where((segID[Y_pred] == bs) & (Y_pred.numpy()>-1))[0]
        J = np.where(segID == bs)[0]
        badcombo = np.nan * np.ones((len(I),len(I)),dtype=np.int64)
        for m1 in range(len(I)):
            for m2 in range(m1+1,len(I)):
                if Y_pred[I[m1]] != Y_pred[I[m2]]:
                    dist = np.linalg.norm(pts[:,I[m1],:] - pts[:,I[m2],:],axis=1)
                    if (~np.isnan(dist)).sum() > 0:
                        if np.nanmean(dist) + np.nanstd(dist) > \
                            segdists['mean'][bs][J==Y_pred[I[m1]].numpy(),J==Y_pred[I[m2]].numpy()] + \
                            3*segdists['std'][bs][J==Y_pred[I[m1]].numpy(),J==Y_pred[I[m2]].numpy()] or \
                            np.nanmean(dist) - np.nanstd(dist) < \
                            segdists['mean'][bs][J==Y_pred[I[m1]].numpy(),J==Y_pred[I[m2]].numpy()] - \
                            3*segdists['std'][bs][J==Y_pred[I[m1]].numpy(),J==Y_pred[I[m2]].numpy()]:
                            badcombo[m1,m2] = 1
                            badcombo[m2,m1] = 1
                        else:
                            badcombo[m1,m2] = 0
                            badcombo[m2,m1] = 0
        for m1 in range(len(I)):
            if (badcombo[m1,:] == 0).sum() == 0 and (badcombo[m1,:]==1).sum() > 0:
                # if no good combos and at least one bad combo,
                confidence_final[I[m1]] = 0
                confidence_weight[I[m1]] = 0
                Y_pred[I[m1]] = -1
    
    # Check for overlapping marker labels and keep label for marker with
    # highest confidence
    for m in range(num_mks):
        visible = (~np.isnan(pts[:,Y_pred==m,0]))
        if (visible.sum(1) > 1).any():
            ii = torch.where(Y_pred==m)[0]
            I = np.argsort(-1*confidence_weight[ii]) # sort by decending confidence
            ii = ii[I]
            # check which ones are actually overlapping
            for j1 in range(len(ii)):
                for j2 in range(j1+1,len(ii)):
                    if Y_pred[ii[j2]] > -1:
                        if ((~np.isnan(pts[:,[ii[j1],ii[j2]],0])).sum(1) > 1).any():
                            # check if these are maybe the same marker due to a 
                            # ghost marker situation
                            d = np.nanmean(np.linalg.norm(pts[:,ii[j1],:] - pts[:,ii[j2],:],axis=1))
                            olframes = ((~np.isnan(np.stack(((pts[:,ii[j1],0]),pts[:,ii[j2],0]),
                                                            axis=1))).sum(1) > 1).sum()
                            if ~(d < 25 or (d < 40 and olframes < 10) or (d < 50 and olframes < 5)):
                                Y_pred[ii[j2]] = -1
                                confidence_final[ii[j2]] = 0
                                confidence_weight[ii[j2]] = 0

    # Attempt to assign labels to unlabelled markers based on probabilities and distances
    unlabelled = torch.where(Y_pred == -1)[0]
    for m in unlabelled:         
        avail_mks = list(set(range(num_mks)) - 
                         set(Y_pred[(~np.isnan(pts[~np.isnan(pts[:,m,0]),:,0])).any(0)].tolist()))
        if len(avail_mks) > 0:
            avail_probs = np.zeros(len(avail_mks))
            for i in range(len(avail_mks)):
                avail_probs[i] = probFrame[m,avail_mks[i],~np.isnan(pts[:,m,0])].mean()
            while avail_probs.max() > 0.1 and Y_pred[m] == -1:
                lbl = avail_mks[np.argmax(avail_probs)]
                segmks = np.where(segID == segID[lbl])[0]
                # Check if distances are withing expected range
                goodcount = 0
                badcount = 0
                for i in range(len(segmks)):
                    if i != np.where(segmks==lbl)[0]:
                        I = torch.where(Y_pred==segmks[i])[0]
                        if I.shape[0] > 0:
                            dist = np.zeros(0)
                            for ii in range(I.shape[0]):
                                dist = np.append(dist,np.linalg.norm(pts[:,m,:] - 
                                                                     pts[:,I[ii],:],axis=1),0)
                            if (~np.isnan(dist)).sum() > 0:
                                if np.nanmean(dist) + np.nanstd(dist) < \
                                    segdists['mean'][segID[lbl]][segmks==lbl,i] + \
                                        3*segdists['std'][segID[lbl]][segmks==lbl,i] and \
                                    np.nanmean(dist) - np.nanstd(dist) > \
                                        segdists['mean'][segID[lbl]][segmks==lbl,i] - \
                                            3*segdists['std'][segID[lbl]][segmks==lbl,i]:
                                    goodcount = goodcount + 1
                                else:
                                    badcount = badcount + 1
                if goodcount > 0:
                    Y_pred[m] = lbl
                    confidence_final[m] = avail_probs.max()
                else:
                    avail_probs[np.argmax(avail_probs)] = 0


    # Attempt to fit segment marker definitions to measured markers based on predicted labels and
    # fill in where there are 3 predicted correctly and one or more unlabelled/incorrect
    y_pred_frame_proposed = np.nan * np.ones((pts.shape[0],pts.shape[1]))
    d_frame_proposed = np.nan * np.ones((pts.shape[0],pts.shape[1]))
    for bs in range(len(uniqueSegs)):
        I = np.where(segID==bs)[0] # label indices for this segment's markers
        if I.shape[0] > 2:
            # Get local coords from marker set
            xsk = np.empty((I.shape[0],3))
            for i in range(I.shape[0]):
                xsk[i,:] = mkcoordL[I[i]]
            xsk = 1000 * xsk
            for k in range(len(keyframes)-1):
                t=keyframes[k]+1
                # Get markers based on predicted labels
                xmk = np.nan * np.ones((I.shape[0],3))
                j = 0
                for i in I:
                    mi = np.logical_and((~np.isnan(pts[t,:,0])),Y_pred.numpy()==i)
                    if mi.sum() == 1:
                        xmk[j,:] = pts[t,mi,:]
                    elif mi.sum() > 1:
                        xmk[j,:] = pts[t,mi,:].mean(0)
                    j = j+1
                
                if (~np.isnan(xmk[:,0])).sum() > 2:
                    # If there are at least 3 visible markers from this segment, use 
                    # Procrustes to line up marker set coords with measured markers                            
                    xsk_g = np.nan * np.ones(xsk.shape)
                    _,xsk_g[~np.isnan(xmk[:,0]),:],T = \
                            procrustes(xmk[~np.isnan(xmk[:,0]),:],xsk[~np.isnan(xmk[:,0]),:])
                    d_mks = np.linalg.norm(xsk_g-xmk,axis=1)
                    
                    # If there is a good fit
                    if np.nanmax(d_mks) < 30:
                        xsk_g = T['scale'] * np.matmul(xsk,T['rotation']) + T['translation']
                        for j in np.where(np.isnan(xmk[:,0]))[0]:
                            d = np.linalg.norm(xsk_g[j,:3] - pts[t,:,:],axis=1)
                            if np.nanmin(d) < 40:
                                y_pred_frame_proposed[keyframes[k]:keyframes[k+1],np.nanargmin(d)] = int(I[j])
                                d_frame_proposed[keyframes[k]:keyframes[k+1],np.nanargmin(d)] = np.nanmin(d)
                    # if there are 4 markers, and all of them but one are less than 30, 
                    # remove label from that one and redo the fitting
                    elif (d_mks[~np.isnan(d_mks)]<30).sum() > 2 and \
                         (d_mks[~np.isnan(d_mks)]>=30).sum() > 0:
                        # Set the presumed incorrect one to nan
                        xmk[np.array([x>=30 if ~np.isnan(x) else False for x in d_mks]),:] = np.nan
                        
                        xsk_g = np.nan * np.ones(xsk.shape)
                        _,xsk_g[~np.isnan(xmk[:,0]),:],T = procrustes(
                                        xmk[~np.isnan(xmk[:,0]),:],xsk[~np.isnan(xmk[:,0]),:])
                        d_mks = np.linalg.norm(xsk_g-xmk,axis=1)
                        if np.nanmax(d_mks) < 30:
                            xsk_g = T['scale'] * np.matmul(xsk,T['rotation']) + T['translation']
                            for j in np.where(np.isnan(xmk[:,0]))[0]:
                                d = np.linalg.norm(xsk_g[j,:3] - pts[t,:,:],axis=1)
                                if np.nanmin(d) < 40:
                                    if np.isnan(y_pred_frame_proposed[t,np.nanargmin(d)]):
                                        y_pred_frame_proposed[
                                            keyframes[k]:keyframes[k+1],np.nanargmin(d)] = int(I[j])
                                        d_frame_proposed[
                                            keyframes[k]:keyframes[k+1],np.nanargmin(d)] = np.nanmin(d)
                                    elif np.nanmin(d) < d_frame_proposed[t,np.nanargmin(d)]: 
                                        y_pred_frame_proposed[
                                            keyframes[k]:keyframes[k+1],np.nanargmin(d)] = int(I[j])
                                        d_frame_proposed[
                                            keyframes[k]:keyframes[k+1],np.nanargmin(d)] = np.nanmin(d)
                                    else:
                                        d[np.nanargmin(d)] = np.nan
                                        if np.nanmin(d) < 40:
                                            if np.isnan(y_pred_frame_proposed[t,np.nanargmin(d)]):
                                                y_pred_frame_proposed[
                                                    keyframes[k]:keyframes[k+1],np.nanargmin(d)] = \
                                                        int(I[j])
                                                d_frame_proposed[
                                                    keyframes[k]:keyframes[k+1],np.nanargmin(d)] = \
                                                        np.nanmin(d)
                                            elif np.nanmin(d) < d_frame_proposed[t,np.nanargmin(d)]:
                                                y_pred_frame_proposed[
                                                    keyframes[k]:keyframes[k+1],np.nanargmin(d)] = \
                                                        int(I[j])
                                                d_frame_proposed[
                                                    keyframes[k]:keyframes[k+1],np.nanargmin(d)] = \
                                                        np.nanmin(d)
    # Find most commonly proposed new label for each trajectory
    y_pred_proposed,_ = stats.mode(y_pred_frame_proposed,axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d_proposed = np.nanmean(d_frame_proposed,axis=0)
    # Go though proposed new labels and see if those will cause overlapping labels
    for m in np.where(~np.isnan(y_pred_proposed[0]))[0]:
        I = Y_pred==int(y_pred_proposed[0][m])
        I[m] = True
        # Make sure this label isn't already used for these frames
        visible = (~np.isnan(pts[:,I,0]))
        if ~(visible.sum(1) > 1).any():
            if Y_pred[m] == -1:
                Y_pred[m] = int(y_pred_proposed[0][m])
            elif d_proposed[m] < 30:
                # Only overwrite an existing label if distance is <30mm
                Y_pred[m] = int(y_pred_proposed[0][m])

    # Get find predicted marker labels
    labels_predicted = []
    for i in range(Y_pred.shape[0]):
        if Y_pred[i] < 0:
            labels_predicted.append('')
        else:
            labels_predicted.append(markers[Y_pred[i]])
    confidence = confidence_final
    
    return labels_predicted, confidence, Y_pred
    
    
def procrustes(X, Y, scaling=True, reflection='best'):
    """
    Source: https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY
    
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0
        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:
        # optimum scaling of Y
        b = traceTA * normX / normY
        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform 

def export_labelled_c3d(pts,labels,rotang,filenamein,filenameout,markers,gapsize):
    '''
    Export a labelled version of the c3d file. Combines markers with the same
    label into a single trajectory and interpolates gaps smaller than 'gapsize'
    using a cubic spline.

    Parameters
    ----------
    pts : numpy array 
        [num_frames x num_markers x 3] array of marker trajectories to be labelled
    labels : list of strings
        predicted labels for each marker trajectory  
    rotang : float
        angle the data was rotated about z by prior to labelling
    filenamein : string
        full path of the original .c3d file that was labelled. This ensures all
        analog data from the original .c3d file is maintained
    filenameout : string
        full path of the file to be exported
    markers : list of string
        labels contained in the marker set
    gapsize : int
        interpolate all gaps in trajectories smaller than this

    Returns
    -------
    None.

    '''

    # recombine marker trajectories
    pts_out = np.empty((pts.shape[0],0,3),dtype=pts.dtype)
    markers_out = []
    for m in range(len(markers)):
        traj = np.nan * np.ones((pts.shape[0],1,3),dtype=pts.dtype)
        I = [i for i,x in enumerate(labels) if x == markers[m]]
        k = 1
        for i in I:
            if ((~np.isnan(np.stack((pts[:,i,0],traj[:,0,0]),axis=1))).sum(1) > 1).any():
                # if there is overlap, then we'll have to add a new marker
                # check if these are maybe the same marker due to a ghost marker situation
                d = np.nanmean(np.linalg.norm(pts[:,i,:] - traj[:,0,:],axis=1))
                olframes = ((~np.isnan(np.stack(((
                                pts[:,i,0]),traj[:,0,0]),axis=1))).sum(1) > 1).sum()
                if ~(d < 25 or (d < 40 and olframes < 10) or (d < 50 and olframes < 5)):
                    markers_out.append(markers[m] + str(k))
                    pts_out = np.append(pts_out,traj,axis=1)
                    traj[:,:,:] = np.expand_dims(pts[:,i,:],axis=1)
                    k = k+1
                else:
                    traj[~np.isnan(pts[:,i,0]),0,:] = np.nanmean(np.stack((
                            pts[~np.isnan(pts[:,i,0]),i,:],
                            traj[~np.isnan(pts[:,i,0]),0,:]),axis=1),axis=1)
            else:
                traj[~np.isnan(pts[:,i,0]),0,:] = pts[~np.isnan(pts[:,i,0]),i,:]
        pts_out = np.append(pts_out,traj,axis=1)
        if k == 1:
            markers_out.append(markers[m])
        else:
            markers_out.append(markers[m] + str(k))
    
    # Fill small gaps
    for m in range(pts_out.shape[1]):
        df = pd.DataFrame(pts_out[:,m,:])
        df = df.interpolate(axis=0,limit=gapsize,limit_area='inside',method='cubic')
        pts_out[:,m,:] = df
              
    # Rotate points back to their original orientation
    Ralign = np.array([[np.cos(rotang),-np.sin(rotang),0],
                       [np.sin(rotang),np.cos(rotang),0],
                       [0,0,1]])
    for i in range(pts_out.shape[1]):
        pts_out[:,i,:] = np.matmul(Ralign.transpose(),
                                   pts_out[:,i,:].transpose()).transpose()
    

    out = c3d(filenamein)
    out['header']['points']['size'] = pts_out.shape[1]
    out['parameters']['POINT']['LABELS']['value'] = markers_out
    out['parameters']['POINT']['DESCRIPTIONS']['value'] = markers_out
    out['parameters']['POINT']['USED']['value'][0] = pts_out.shape[1]
    out['parameters']['POINT']['FRAMES']['value'][0] = pts_out.shape[0]
    out['data']['points'] = np.ones((4,pts_out.shape[1],pts_out.shape[0]),
                                    dtype=pts_out.dtype)
    out['data']['points'][0:3,:,:] = pts_out.transpose((2,1,0))
    out['data']['meta_points']['residuals'] = np.zeros((1,pts_out.shape[1],
                                                 pts_out.shape[0]),dtype=pts_out.dtype)
    out['data']['meta_points']['camera_masks'] = np.zeros((
                out['data']['meta_points']['camera_masks'].shape[0],
                pts_out.shape[1],pts_out.shape[0]),dtype=bool)
    out.write(filenameout)