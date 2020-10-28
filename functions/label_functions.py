# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:12:11 2020

Functions to label motion capture markers using the automatic labelling 
algorithm.

@author: aclouthi
"""

import pickle
import copy
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.utils.extmath import weighted_mode
from scipy import signal
from scipy import stats
from scipy.optimize import linear_sum_assignment  
from ezc3d import c3d

import functions.import_functions as io 
import functions.nn_functions as nnf

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
    markers, segment, uniqueSegs, segID, mkcoordL, num_mks = io.import_markerSet(markersetpath)
    
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
    windowIdx = io.window_data([pts],windowSize,pts.shape[1])
    
    # Apply neural network to get label probablities within windows
    probWindow = nnf.predict_nn(modelpath,pts,windowIdx,scaleVals,num_mks,max_len)
    
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