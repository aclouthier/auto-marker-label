# An Open Source Algorithm for the Automatic Labelling of Motion Capture Markers using Deep Learning

An algorithm that uses machine learning to automatically label optical motion capture markers. The algorithm can be trained on existing data or simulated marker trajectories. Data and code is provided to generate the simulated trajectories for custom marker sets.
 
![Marker Labelling GUI](/images/auto-marker-label-GUI.jpg) 
 
## Installation
This code has been tested using Python 3.7. The following packages are required and can be installed using pip. The version used in testing is listed. See requirements.txt for the full list.

dash==1.14.0
dash-bootstrap-components==0.10.3
dash-core-components==1.10.2
dash-html-components==1.0.3
mydcc==0.1.22
numpy==1.19.1
pandas==1.0.5
plotly==4.9.0
scipy==1.5.2
torch==1.5.1

[ezc3d](https://github.com/pyomeca/ezc3d) is also required and can be installed using conda
'conda install -c conda-forge ezc3d'

## Use
### Generate Simulated Trajectories
If you do not have existing labelled motion capture data to train the algorithm, simulated trajectories can be generated. 

If you have your own training data, skip this step.
- First, the marker set to be used must be defined using an OpenSim marker set .xml file. 
  - Install [OpenSim](https://simtk.org/frs/index.php?group_id=91)
  - In OpenSim, Open Model Rajagioa2015_mod.osim included in /data folder
  - Right click on Markers in the Navigator to add new marker. Marker can be selected and moved in the visualization. Set the marker's name and parent_frame (ie body segment it is attached to) in the Properties window.
  - Save the marker set by right clicking Markers in the Navigator and choosing Save to File.
- Set parameters in **generateSimTrajectories.py**
  - Set the path to the marker set .xml file. 
  - Set the markers used to align the data. Ideally, these will be a marker on the left and right side of the torso or head (eg. right and left acromions). These are used to rotate the data about the verical axis so that the person faces the +x direction.
- Run **generateSimTrajectories.py**

### Train the Algorithm
- Set parameters in **trainAlgorithm.py**.
  - If using simulated trajectories to train, set the path to the marker set and to the pickle file generated by generateSimTrajectories.py.
  - If using existing labelled data to train, set the path to the folder containing the c3d files. Set the markers used to align the data. Ideally, these will be a marker on the left and right side of the torso or head (eg. right and left acromions).
- Run **trainAlgorithm.py**. The training will be performed on a GPU, if one is available. 
Note that this may take a long time to run (ie. hours - days). Training time can be reduced by using less training data (set num_participants in generateSimTrajectories.py).

### Setup the GUI
- Set the paths to trained model, trained values pickle file, and market set in **markerLabelGUI.py**.
- Run **markerLabelGUI.py**, this will open the GUI in your browser.

### Using the GUI 
- Enter the folder path where the c3d files to be labelled are located.
- Select the desired file from the dropdown menu.
- Click *Load Data*, the loading is complete when the plot of markers appears.
- If the person is not facing the +x direction, enter the angle to rotate the data about the z-axis (vertical) and click *Submit*. This angle should be chosen such that the person faces +x for the majority of the trial.
- Click *Label Markers*.
- Examine the results.
  - Clicking and dragging rotates, the scroll wheel zooms, and right clicking translates the plot. Hovering over a marker displays the marker number, label, and coordinates.
  - The slider at the bottom can be used to view different time frames. After clicking the slider, the left and right arrow keys can be used to adjust the timeframe as well.
  - The type of marker visualization can be selected from the *Visualization* dropdown menu. *Confidence* colours the markers based on the confidence in the predicted label. *Unlabelled* highlights unlabelled markers in red. *Segments* displays markers attached to the same body segment in the same colour.
  - The *Error Detection* box lists unlabelled markers and duplicated labels. The time frames where the error was detected are displayed. Note that the listed marker is guaranteed to be visible for the first and last frames, but may be missing from the intermediate frames of the listed range.
- Correct incorrect labels using the *Marker Label Modifier*. Type the number for the marker to change the label and select the desired label from the dropdown then click the *Submit* button. To leave a marker unlabelled, leave the dropdown menu blank (this can be cleared by clicking the 'x').
- Export a labelled version of the .c3d file by clicking the *Export to C3D* button. This will rotate the data back to the original orientation and fill small gaps through cubic spline interpolation. Unlablled markers will not be exported.
- Before loading a new c3d file, click the *Refresh Settings* button.

### Transfer Learning
- As data is collected, labelled, and corrected, it can be added to the training set through transfer learning to improve the algorithm
- In **transferLearning.py**, set the path for the trained model and training values to build on (.ckpt and .pickle file) and the folder containing the .c3d files to add to the training set. Set the markers used to align the data. Ideally, these will be a marker on the left and right side of the torso or head (eg. right and left acromions).
- Run **transferLearning.py**
