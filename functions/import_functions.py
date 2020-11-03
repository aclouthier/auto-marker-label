# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:37:13 2020

import_functions.py

Functions used to import data.

@author: aclouthi@uottawa.ca
"""

import xml.dom.minidom
import numpy as np
import torch
import pandas as pd
from scipy import signal
from ezc3d import c3d

filtfreq = 6 # Cut off frequency for low-pass filter for marker trajectories



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
                                windowIdx.append([t,m,i1,i2])
                            if  (~np.isnan(pts[i2:,m,0])).sum() > 1: # any more visible markers?
                                i1 = i2 + np.where(~np.isnan(pts[i2:,m,0]))[0][0]
                            else: 
                                i1 = pts.shape[0] + 1
                        else:
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
    fs : int
        Sampling frequency used in .c3d files
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
        fs = c3ddat['parameters']['POINT']['RATE']['value']
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
    
    # Try to find and fix places where the markers swap indices
    thresh = 20
    for m in range(rawpts.shape[1]):
        kf = np.where(np.isnan(rawpts[1:,m,0]) != np.isnan(rawpts[0:-1,m,0]))[0]
        if ~np.isnan(rawpts[0,m,0]):
            kf = np.insert(kf,0,-1,axis=0)
        if ~np.isnan(rawpts[-1,m,0]):
            kf = np.concatenate((kf,[rawpts.shape[0]-1]))
        kf = np.reshape(kf,(-1,2))
        k = 0
        while k < kf.shape[0]-1:
            d = np.linalg.norm(rawpts[kf[k+1,0]+1,m,:] - rawpts[kf[k,1],m,:])
            all_d = np.linalg.norm(rawpts[kf[k,1]+1,:,:] - rawpts[kf[k,1],m,:],axis=1)
            all_d[m] = np.nan
            if (~np.isnan(all_d)).sum() > 0:
                if d > np.nanmin(all_d) and np.nanmin(all_d) < thresh and \
                        np.isnan(rawpts[kf[k,1],np.nanargmin(all_d),0]):
                    dummy = rawpts[kf[k,1]+1:,m,:].copy()
                    rawpts[kf[k,1]+1:,m,:] = rawpts[kf[k,1]+1:,np.nanargmin(all_d),:]
                    rawpts[kf[k,1]+1:,np.nanargmin(all_d),:] = dummy.copy()
                    
                    kf = np.where(np.isnan(rawpts[1:,m,0]) != np.isnan(rawpts[0:-1,m,0]))[0]
                    if ~np.isnan(rawpts[0,m,0]):
                        kf = np.insert(kf,0,0,axis=0)
                    if ~np.isnan(rawpts[-1,m,0]):
                        kf = np.concatenate((kf,[rawpts.shape[0]-1]))
                    kf = np.reshape(kf,(-1,2))
            k = k+1
            
    # Wherever there is a gap, check if the marker jumps further than the distance to the 
    # next closest marker. If so, split it into a new trajectory.
    pts = np.empty((rawpts.shape[0],0,3))
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
                

    # Angle to rotate points about z-axis
    rotang = float(rotang) * np.pi/180 
    Ralign = np.array([[np.cos(rotang),-np.sin(rotang),0],
                       [np.sin(rotang),np.cos(rotang),0],
                       [0,0,1]])
    for i in range(pts.shape[1]):
        pts[:,i,:] = np.matmul(Ralign,pts[:,i,:].transpose()).transpose()
                
    return pts, fs