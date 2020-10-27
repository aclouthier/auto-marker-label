# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:05:25 2020

Functions to train marker labelling algorithm and apply neural network results.

Please cite: 

@author: aclouthi@uottawa.ca
"""

import numpy as np
import torch
import torch.nn as nn
import random
import copy

# Neural network architecture parameters
batch_size = 100
nLSTMcells = 256
nLSTMlayers = 3
LSTMdropout = .17
FCnodes = 128
# Learning parameters
lr = 0.078
momentum = 0.65

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