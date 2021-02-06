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
import automarkerlabel as aml

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
num_epochs = 10 # Number of epochs to train for (default=10)
min_loss = 0.05 # Minimum loss expected for each step (default=None)
# Path to a .ckpt file of a previously trained neural network if using transfer learning.
# Set to None if not using a previous model.
prevModel = None 

# --- Parameters required if using existing labelled c3ds as training data --- #
windowSize = 120  # Desired size of data windows (default=120)
# Markers to use to align person such that they face +x. Suggest acromions or pelvis markers.
alignMkR = 'RAC'
alignMkL = 'LAC'
# --------------------------------------------------------------------------- #

# Run training
aml.trainAlgorithm(fld,datapath,markersetpath,fs,num_epochs,min_loss,prevModel,windowSize,
                   alignMkR,alignMkL)
