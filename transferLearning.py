# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:11:16 2020

Use this script to perform transfer learning. Requires a previously trained model and
labelled c3d files to add to training set.

@author: aclouthi
"""

import os
import automarkerlabel as aml

# ---------------------------- INPUT PARAMETERS ----------------------------- #

num_epochs = 10 # Number of epochs to train for
min_loss = 0.05 # Minimum loss expected for each step (default=None)
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

# --------------------------------------------------------------------------- #


aml.transferLearning(fld,datapath,modelpath,trainvalpath,markersetpath,num_epochs,
                 min_loss,windowSize,alignMkR,alignMkL)

