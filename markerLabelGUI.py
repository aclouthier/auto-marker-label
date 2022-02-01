import dash
from dash import Dash #to be installed
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc #to be installed
import plotly.graph_objects as go #to be installed
from json import JSONEncoder
import mydcc #to be installed
import numpy as np
import os
import webbrowser
import logging
import glob
import time

import automarkerlabel as aml

path=os.path.dirname(os.path.abspath(__file__)) # get current path to use as default

# --------------------------------------------------------------------------- #
# --------------------------------- PARAMETERS ------------------------------ #
# --------------------------------------------------------------------------- #

# File paths
modelpath = os.path.join(path,'data','model_sim_add9_2020-09-15.ckpt')
trainvalpath = os.path.join(path,'data','trainingvals_sim_add9_2020-09-15.pickle')
markersetpath = os.path.join(path,'data','MarkerSet.xml')

# Other
gapfillsize = 24 # Size of gaps to fill with interpolated data when exporting
windowSize = 120 # size of windows used to segment data for algorithm
incr=1 # increment for GUI slider

# --------------------------------------------------------------------------- #

logging.getLogger('werkzeug').setLevel(logging.ERROR)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    


c3dlist = glob.glob(os.path.join(path,'*.c3d'))

# Read marker set
markers, segment, uniqueSegs, segID, _, num_mks = aml.import_markerSet(markersetpath)

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets, 
           title='Marker Labelling GUI', update_title='GUI UPDATING...')

# Define Layout
app.layout = html.Div(
    [
        mydcc.Relayout(id = "camera", aim = 'the_graph' ),
        dbc.Row(dbc.Col(html.H1(children= "3D Visualization of Markers"))),
        
        dbc.Col([dbc.Row([
            dbc.Col(html.H2("MODIFY LABELS AND SPLIT TRAJECTORIES :")),
            dbc.Col(html.H2("ERROR DETECTION :")),
            dbc.Col(html.H2("IMPORT DATA FILE :"))]),
                dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(
                        html.Div(dcc.Input(
                        id='marker_ind',
                        className='entry_button',
                        placeholder='Current marker #',
                        type='text',
                        value='',
                        style={'margin-top':'6px'}
                        )),width=4),
                    dbc.Col(
                        html.Div(dcc.Dropdown(
                        id='new_name',
                        className='dropdown',
                        placeholder='New name',
                        options=[{'label': j, 'value': j} for j in markers]
                        )), width=4),
                    dbc.Col(width=1),
                    dbc.Col(
                        html.Button(
                        id='submit-button', 
                        className='submit_button',
                        type='submit', 
                        children='Rename',
                        style={'margin-bottom':'5px'}), width=3)]),
                dbc.Row([
                    dbc.Col(html.Div('Split marker trajectory:',
                                     style={'text-align':'centre','width': '100%','margin-top': '2px'}),width=4),
                    dbc.Col(
                        html.Div(dcc.Input(
                        id='traj_split_frame',
                        className='entry_button',
                        placeholder='Frame # to split at',
                        type='text',
                        value='',
                        style={'margin-top':'6px'}
                        )),width=5),
                    # dbc.Col(width=1),
                    dbc.Col(
                        html.Button(
                        id='submit-split-button', 
                        className='submit_button',
                        type='submit', 
                        children='Split',
                        style={'margin-top':'5px'}), width={'size':3, 'offset':0})
                    ]),

                ]),
            
            dbc.Col(html.Div(id='errors',
                             className='errors'), width=4),
            
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.Div("Click before loading new file:", 
                              style={'text-align':'left','width': '100%','margin-bottom': '5px'}),
                              width=6,align='center'),
                    dbc.Col(html.A(html.Button('Refresh\nSettings', 
                                               id='refresh', 
                                               className='refresh'), href='/'),
                            width=6)
                    
                    ]),
                dbc.Row([
                    dbc.Col(html.Div('Enter data folder:',
                                      style={'text-align' : 'left','width': '80%',
                                             'margin-bottom':'3px'}),
                            width=6,align='center'),
                    dbc.Col(html.Div(dcc.Input(
                            id='datapath',
                            className='datapath',
                            type='text',
                            placeholder=path,
                            debounce=True,
                            size=35,
                            persistence=True,
                            style={'width': '100%','margin-bottom':'5px'})),
                        width=6)
                    ]),
                dbc.Row([
                    dbc.Col(html.Div(
                            dcc.Dropdown(
                            id='upload_data',
                            className='dropdown2',
                            placeholder='Select File',
                            value=None,
                            options=[{'label': j.split(os.path.sep)[-1], 'value': j, 
                                      'title': j.split(os.path.sep)[-1]} for j in c3dlist])),
                        width=6,align='center'),
                    dbc.Col(html.Button(
                            id='load_data_button',
                            className='submit_button',
                            type='submit',
                            children='Load Data',
                            style={'width':'100%','border-width':'2px'}),
                        width=6)
                    ])
                ],width=4)
            ])
            ]),
     

        dbc.Col([dbc.Row([
            dbc.Col(html.Div(id='comment'), width=4),
            dbc.Col(html.Div(""),width=4),
            dbc.Col(html.Div(id='outputdata', className='output_data'), width=4)
            ])
            ]),
        
        dbc.Col([dbc.Row([
            dbc.Col(html.H2("VISUALIZATION :")),
            dbc.Col(html.H2("ROTATION ABOUT Z :")),
            dbc.Col(html.H2("LABEL MARKERS:"))]),
                    dbc.Row([
                dbc.Col(html.Div(dcc.Dropdown(
                    id='dropdown',
                    className='dropdown',
                    options=[{'label': 'Confidence', 'value': 'Confidence'},
                             {'label': 'Unlabelled','value': 'Unlabelled'},
                             {'label': 'Segments','value':'Segments'}],
                    value='Confidence'
                    ),
                    style={"margin":"0px 0px 0px 0px"}), width = {'size':2, 'offset':1}),
                dbc.Col(html.Div(dcc.Input(
                    id='rotation',
                    className='entry_button',
                    placeholder='Enter rotation angle',
                    type='text',
                    value=0
                    )), width={'size':2, 'offset':1}),
                dbc.Col(html.Div(id='rot_comment'), width=1),
                dbc.Col(html.Button(
                    id='submit-button-rot', 
                    className='submit_button',
                    type='submit', 
                    children='Submit',
                    style={'height':'50%'}
                    ), width={'size':1, 'offset':0}),
                dbc.Col([dbc.Row(html.Button(
                    id='load_button',
                    className='submit_button',
                    type='button',
                    children='Label Markers!',
                    style={
                        'margin':'0px 0px 0px 0px'
                        })),
                    dbc.Row(html.Div(id='label_comment'))], width={'size':2, 'offset':1})
                ]),
                    ]),
                    

        dbc.Col([dbc.Row([
            dbc.Col(html.H2("MARKER LABELS LIST :"), width=4),
            dbc.Col(html.H2("3D PLOT :"), width=8)]),
                    dbc.Row([        
                dbc.Col(html.Div(id='labels_list_current', className='labels_list',
                                 style={'margin': '10px 0px 0px 10px'}), width=4),
                dbc.Col(dcc.Loading(
                    id="loading_graph",
                    children=[html.Div(dcc.Graph(id='the_graph', 
                                                 style={'height':'50vh', 'margin-left':'20%'}))],
                    type="circle"), width=8),
            ]),
        ]),
        
        
        dbc.Row(dbc.Col(html.Div(id='timeframe'), 
                        style={'margin': '-10px 0px 0px 0px','position':'top'})),
        dbc.Row([
            dbc.Col(html.Div(dcc.Slider(
                id="Time_Slider",
                min=0,
                value=1,
                step=incr,
                updatemode="drag"),
                style={"position":"left"}), width=11),
            dbc.Col(html.Button(
                    id='export_button',
                    className='export_button',
                    type='button',
                    children='EXPORT TO C3D'), width=1)]),
        
        dbc.Row(dbc.Col(html.Div(id='export_comment')),style={'margin-top':'5px'}),
        dbc.Row(dbc.Col(html.Div(id='pts_c3d'),style={'display':'none'})),
        dbc.Row(dbc.Col(html.Div(id='labels_c3d'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='labels_updated'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='confidence_c3d'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='body_segment'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='frame_rate'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='rotang'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='c3dlist'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='filename'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='rawlabels'), style={'display': 'none'})),
        dbc.Row(dbc.Col(html.Div(id='pts_updated'),style={'display':'none'})),
        dbc.Row(dbc.Col(html.Div(id='pts_current'),style={'display':'none'})),
        dbc.Row(dbc.Col(html.Div(id='pts_rot'),style={'display':'none'})),
        dbc.Row(dbc.Col(html.Div(id='confidence_current'),style={'display':'none'})),
        dbc.Row(dbc.Col(html.Div(id='pts_split'),style={'display':'none'})),
        dbc.Row(dbc.Col(html.Div(id='labels_split'),style={'display':'none'})),
        dbc.Row(dbc.Col(html.Div(id='confidence_split'),style={'display':'none'})),   
        dbc.Row(dbc.Col(html.Div(id='labels_current'),style={'display':'none'})),    
        dbc.Row(dbc.Col(html.Div(id='split_timestamp'),style={'display':'none'})),    
        dbc.Row(dbc.Col(html.Div(id='mod_timestamp'),style={'display':'none'})),    
        dbc.Row(dbc.Col(html.Div(id='rot_timestamp'),style={'display':'none'})), 
        dbc.Row(dbc.Col(html.Div(id='label_timestamp'),style={'display':'none'})), 
        dbc.Row(dbc.Col(html.Div(id='message_mod'),style={'display':'none'})),    
        dbc.Row(dbc.Col(html.Div(id='message_split'),style={'display':'none'})),    
    ]
    )

#Update file list dropdown
@app.callback(Output('upload_data','options'),
              [Input('datapath','value')])

def update_filelist(datapath):
    print('updating file list')
    if datapath is not None:
        c3dlist = glob.glob(os.path.join(datapath,'*.c3d'))
    else:
        c3dlist = glob.glob(os.path.join(path,'*.c3d'))
    return [{'label': j.split(os.path.sep)[-1], 'value': j} for j in c3dlist]


# Submit rotation angle
@app.callback([Output('rotang','children'), Output('rot_comment','children'),
               Output('rot_timestamp', 'children'), Output('pts_rot', 'children')],
              [Input('submit-button-rot','n_clicks')],
              [State('rotation','value'), State('pts_current', 'children'), State('rotang', 'children')])

def submit_angle(n_clicks, angle, pts, rotang_old):
    
    comment='Rotation : ' + str(angle)
    print('rotating')
    if n_clicks is not None:
        if rotang_old is None:
            rotang_old=0
        pts=np.array(pts)
        pts=pts.astype('float64')
        timestamp=time.time()
        rotang = float(angle) * np.pi/180 
        rotang_current=rotang_old+rotang
        Ralign = np.array([[np.cos(rotang),-np.sin(rotang),0], 
                       [np.sin(rotang),np.cos(rotang),0],
                       [0,0,1]])
        for i in range(pts.shape[1]):
            pts[:,i,:] = np.matmul(Ralign,pts[:,i,:].transpose()).transpose()
            
    elif n_clicks is None:
        raise PreventUpdate
    
    return rotang_current, comment, timestamp, pts

# Load data button
@app.callback(Output('upload_data','value'),
              [Input('load_data_button','n_clicks')],
              [State('upload_data','value')])

def submit_filename(n_clicks,filename):
    if n_clicks is not None:
        return filename
    else:
        return dash.dash.no_update

#Load data 
@app.callback(
    [Output('outputdata','children'),Output('pts_c3d','children'),
     Output('frame_rate', 'children'),Output('Time_Slider','max'),
     Output('Time_Slider', 'marks'),Output('rawlabels','children')],
    [Input('rotang', 'children'),Input('load_data_button','n_clicks')],
    [State('upload_data','value')])

def load_data(angle,n_clicks,filename):
    
    outputdata=''
    pts=''
    fs=''
    max_slider=''
    marks=''
    labels = ''

    if angle is None:
        angle=0

    if (filename is not None):
        if os.path.exists(filename) and ('.c3d' in filename):
            outputdata=filename
            
            print('importing points')
    
            # angle=0+int(angle)
            angle=float(angle)
            pts_c3d, fs, labels = aml.import_raw_c3d(filename, angle)
            
            pts = np.array(pts_c3d, dtype=np.float64)
            
            #Slider tick marks
            max_slider=len(pts)
            if max_slider < 350:
                step = 5
            elif max_slider < 700:
                step = 10
            else:
                step = round(max_slider/700) * 10
            marks_slider = list(range(0,len(pts),step))
        
            marks={str(marks_slider[i]):{'label':str(marks_slider[i]), 
                             "style":{'font-size':'1vw','writing-mode': 'vertical-rl',
                             'text-orientation': 'mixed'}} for i in range(0,len(marks_slider))} 

        elif os.path.exists(filename) and ('.c3d' not in filename):
            outputdata = 'ERROR: File must be in .c3d format'
        else:
            outputdata = 'ERROR: file not found \n%s' % (filename)

    return outputdata, pts, fs, max_slider, marks, labels

# Label Data
@app.callback([Output('labels_c3d', 'children'), Output('confidence_c3d', 'children'), 
               Output('body_segment', 'children'), Output('label_comment', 'children'),
               Output('label_timestamp', 'children')],
              [Input('load_button', 'n_clicks')], [State('pts_current', 'children'),
               State('frame_rate', 'children'), State('rawlabels','children'),
               State('upload_data', 'value')])

def label_data(n_clicks, pts, fs, rawlabels, filename):
        
    labels=''
    confidence=''
    body_segment=''
    comment=''
    timestamp=None
    if (filename is not None) and (pts != ''):

        pts=np.array(pts)
        
        if n_clicks is None:
            labels= rawlabels #[0]*(len(pts[0,:,0]))
            confidence= [0]*(len(pts[0,:,0]))
            body_segment=None
            comment="Labels not assigned"
        elif n_clicks is not None:        
            print('finding labels and confidence scores')
            timestamp=time.time()
            labels_c3d, confidence_c3d,_ = aml.marker_label(pts,modelpath,trainvalpath,
                                                           markersetpath,fs,windowSize)
            print('done labelling')
            labels=np.array(labels_c3d)
            confidence=np.array(confidence_c3d)
            body_segment=1
            comment="Labels assigned"
                                
    return labels, confidence, body_segment, comment, timestamp

# Marker label modifier
@app.callback([Output('message_mod','children'),Output('labels_updated','children'), Output('mod_timestamp', 'children')],
              [Input('submit-button','n_clicks')],
              [State('labels_current', 'children'),State('marker_ind', 'value'),State('new_name', 'value'), State('split_timestamp', 'children'), State('mod_timestamp', 'children')])

def name_mod(n_clicks, labels, marker_ind, new_name, split_timestamp, mod_timestamp):
    
    ctx = dash.callback_context
    message=""
    
    if labels is not None:
        print('listing labels')
        if n_clicks is None:
            labels_updated=np.array(labels)
        
        if ctx.triggered:
            if n_clicks is not None:
                labels_updated=np.array(labels)
                mod_timestamp=time.time()
                index = int(marker_ind)-1 
                labels_updated[index]=str(new_name)
                message='Marker #"{}" has been labelled as "{}"'.format(marker_ind, str(new_name))
    else:
        raise PreventUpdate
    
    return message, labels_updated, mod_timestamp

# Split trajectory
@app.callback([Output('pts_split','children'),Output('labels_split', 'children'), 
                Output('confidence_split', 'children'), Output('split_timestamp', 'children'),
                Output('message_split', 'children')],
              [Input('submit-split-button', 'n_clicks')],
              [State('pts_current', 'children'), State('confidence_current','children'),
                State('labels_current','children'), State('traj_split_frame','value'),State('marker_ind', 'value')])

def trajectory_split(n_clicks,pts,confidence,labels,split_idx,marker_ind):
    
    print('splitting')
    timestamp=time.time()
    message=''
    
    if n_clicks is not None:
        pts=np.array(pts)
        mk_idx = int(marker_ind) - 1
        pts = np.concatenate((pts,np.expand_dims(pts[:,mk_idx,:],1)),axis=1) #Duplicate selected marker
        pts[int(split_idx):,mk_idx,:] = np.nan
        pts[:int(split_idx),-1,:] = np.nan
        
        pts_split=pts
        confidence=np.array(confidence)
        confidence = np.append(confidence, confidence[mk_idx])
        confidence_split=confidence
        
        labels_split = np.append(labels,labels[mk_idx])
        message='Marker #"{}" has been split at timeframe "{}"'.format(marker_ind, str(split_idx))
    elif n_clicks is None:
        raise PreventUpdate
    
    return pts_split, labels_split, confidence_split, timestamp, message 

# Label variables handler
@app.callback([Output('labels_current','children'), Output('labels_list_current', 'children'),
               Output('comment', 'children')],
               [Input('labels_split','children'), Input('labels_c3d', 'children'), Input('labels_updated', 'children'),
                Input('rawlabels', 'children')], 
               [State('split_timestamp', 'children'), State('mod_timestamp', 'children'), State('rot_timestamp', 'children'),
                State('label_timestamp', 'children'), State('message_mod', 'children'), State('message_split', 'children')])

def label_handler(labels_split, labels_c3d, labels_updated, rawlabels, split_timestamp, mod_timestamp, rot_timestamp, label_timestamp, message_mod, message_split):
    
    print('Selecting current label variable')
    labels_current=labels_c3d #labels from initial importation
    labels_list_current=''
    comment=''
    
    if split_timestamp is None:
        split_timestamp=0
    if mod_timestamp is None:
        mod_timestamp=0    
    if rot_timestamp is None:
        rot_timestamp=0    
    if label_timestamp is None:
        label_timestamp=0  
        
    timestamps=[split_timestamp, mod_timestamp, rot_timestamp, label_timestamp]
    timestamps=np.array(timestamps)
    
    if label_timestamp==0 and rawlabels: #Markers still unlabelled
        labels_current= rawlabels 
    if np.nanmax(timestamps)==0:
        # print('NOTHING DONE YET')
        if len(labels_current)>1:
            for j in range(len(labels_current)):
                labels_list_current+= str(j+1) + ". " + str(labels_current[j])+'\n'
            
        return labels_current, labels_list_current, comment
    
    if np.nanmax(timestamps)==split_timestamp: #Most recent is SPLIT (last used)
        # print('SPLIT LAST USED-labels')
        labels_current=labels_split #use labels_split instead of labels_updated
        comment=message_split
    elif np.nanmax(timestamps)==mod_timestamp:
        # print('MOD LAST USED-labels')
        labels_current=labels_updated
        comment=message_mod
    elif np.nanmax(timestamps)==rot_timestamp:
        raise PreventUpdate
        
    if len(labels_current)>1:
        for j in range(len(labels_current)):
            labels_list_current+= str(j+1) + ". " + str(labels_current[j])+'\n'
    
    return labels_current, labels_list_current, comment            
    
#Pts/confidence variable handler
@app.callback([Output('pts_current','children'), Output('confidence_current', 'children')],
        [Input('pts_c3d', 'children'), Input('pts_split','children'), 
         Input('confidence_c3d', 'children'), Input('confidence_split', 'children'),
         Input('pts_rot', 'children')],
        [State('split_timestamp', 'children'), State('mod_timestamp', 'children'),
         State('rot_timestamp', 'children'), State('label_timestamp', 'children')])

def pts_handler(pts_c3d, pts_split, confidence_c3d, confidence_split, pts_rot, split_timestamp, mod_timestamp, rot_timestamp, label_timestamp):
    
    print('Selecting current pts variable')
    pts_current=pts_c3d #pts from initial importation
    confidence_current=confidence_c3d
    
    if split_timestamp is None:
        split_timestamp=0
    if mod_timestamp is None:
        mod_timestamp=0    
    if rot_timestamp is None:
        rot_timestamp=0    
    if label_timestamp is None:
        label_timestamp=0    
    
    timestamps=[split_timestamp, mod_timestamp, rot_timestamp, label_timestamp]
    timestamps=np.array(timestamps)

    if label_timestamp==0 and pts_current: #Markers still unlabelled
        # print('MARKERS NOT LABELLED')
        pts_temp=np.array(pts_current)
        confidence_current=[0]*(pts_temp.shape[1])
        
    if np.nanmax(timestamps)==0:
        return pts_current, confidence_current
    
    if np.nanmax(timestamps)==label_timestamp: #Most recent is LABEL MARERS (last used)
        # print('LABELLING LAST USED')
        confidence_current=confidence_c3d
        return dash.dash.no_update, confidence_current
    if np.nanmax(timestamps)==split_timestamp: #Most recent is SPLIT (last used)
        # print('SPLIT LAST USED-pts')
        pts_current=pts_split
        confidence_current=confidence_split
    elif np.nanmax(timestamps)==rot_timestamp:
        # print('ROT LAST USED-pts')
        pts_current=pts_rot
        confidence_current=confidence_c3d
        return pts_current, dash.dash.no_update
    
    return pts_current, confidence_current    

# Error list
@app.callback(Output('errors', 'children'),
              [Input('labels_current', 'children'), Input('pts_current','children')],
              [State('split_timestamp', 'children'), State('mod_timestamp', 'children')])

def update_error(labels,pts, split_timestamp, mod_timestamp):
    errors=''
    
    if split_timestamp is None:
        split_timestamp=0
    if mod_timestamp is None:
        mod_timestamp=0    
  
    split_timestamp=float(split_timestamp)
    mod_timestamp=float(mod_timestamp)
    
    if split_timestamp>mod_timestamp: #Most recent is SPLIT (last used)
        # print('SPLIT LAST USED')
        raise PreventUpdate
        
    if pts != '':
        print('checking errors')
        errors='Label errors have been found for the following markers : \n'
        pts = np.array(pts, dtype=np.float64)
        #duplicates index
        index_duplicates = []
        overlap_frames = []
        for m in range(len(markers)):
            ii = [i for i,x in enumerate(labels) if x == markers[m]]
            visible = (~np.isnan(pts[:,ii,0]))
            if (visible.sum(1) > 1).any():
                # check which ones are actually overlapping
                for j1 in range(len(ii)):
                    for j2 in range(j1+1,len(ii)):
                        if labels[ii[j2]] != '':
                            if ((~np.isnan(pts[:,[ii[j1],ii[j2]],0])).sum(1) > 1).any():
                                index_duplicates.append(ii[j1])
                                index_duplicates.append(ii[j2])
                                overlap_frames.append('%d - %d' % 
                                               (np.where((~np.isnan(
                                                pts[:,[ii[j1],ii[j2]],0])).sum(1)>1)[0].min(),
                                                np.where((~np.isnan(
                                                pts[:,[ii[j1],ii[j2]],0])).sum(1)>1)[0].max()))
                                overlap_frames.append('%d - %d' % 
                                               (np.where((~np.isnan(
                                                pts[:,[ii[j1],ii[j2]],0])).sum(1)>1)[0].min(),
                                                np.where((~np.isnan(
                                                pts[:,[ii[j1],ii[j2]],0])).sum(1)>1)[0].max()))
        
        nbr=0 # number of errors
        for i in range(0,len(labels)):
            if i in index_duplicates:
                errors+= 'DUPLICATED: '+ str(i+1) + '. '  + \
                            labels[i]+ ', frames '+ \
                            overlap_frames[index_duplicates.index(i)]+'\n'
                nbr=nbr+1
            if labels[i] not in markers and labels[i] != '' and \
                    labels[i] != 0 and labels[i] != 'None': 
                errors+= 'MISPELLED: '+ str(i+1) + '. ' + labels[i] +'\n'
                nbr=nbr+1
            if labels[i] == '' and (~np.isnan(pts[:,i,0])).sum() > 0:
                errors += 'UNLABELLED: ' + str(i+1) + '. frames ' + \
                    str(np.where((~np.isnan(pts[:,i,0])))[0].min()) + ' - ' + \
                    str(np.where((~np.isnan(pts[:,i,0])))[0].max()) + '\n'
                nbr = nbr+1
        
        if nbr==0:
            errors='No errors have been found.'
        errors='('+str(nbr)+')'+errors
        
    return errors

#Timeframe indicator
@app.callback(Output('timeframe','children'),
    [Input('Time_Slider', 'value')])

def current_timeframe(value):
    return 'Timeframe: "{}"'.format(value);

           
#Slider widget & Dropdown
@app.callback(Output('the_graph','figure'),
             [Input('dropdown', 'value'), Input('Time_Slider', 'value'), 
              Input('pts_current','children'), Input('labels_current', 'children'),
              Input('confidence_current','children')])

def update_graph(dropdown,Time_Slider, pts, labels, confidence):
            
    fig=''

    if pts != '':
        print('graphing')
        
        pts= np.array(pts, dtype=np.float64)
        confidence=np.array(confidence)
              
        labels_num = []
        for i in range(len(labels)):
            labels_num.append('%d. %s' % (i+1,labels[i]))
        labels=np.array(labels)
        labels_num=np.array(labels_num)
        
        #Define points
        X = pts[:,:,0] 
        Y = pts[:,:,1] 
        Z = pts[:,:,2] 
        
        #Center & Scale Axis
        max_range = np.array([np.nanmax(X)-np.nanmin(X), np.nanmax(Y)-np.nanmin(Y), 
                              np.nanmax(Z)-np.nanmin(Z)]).max() / 2.0 
        
        mid_x = (np.nanmax(X)+np.nanmin(X)) * 0.5 #middle value x, not necessarily 0
        mid_y = (np.nanmax(Y)+np.nanmin(Y)) * 0.5 #middle value y
        mid_z = (np.nanmax(Z)+np.nanmin(Z)) * 0.5 #middle value z
        
        
        if dropdown=='Confidence':
            fig = go.FigureWidget(
                data=[
                go.Scatter3d(
                    x=pts[Time_Slider,:,0],
                    y=pts[Time_Slider,:,1],
                    z=pts[Time_Slider,:,2],
                    showlegend=False,
                    visible=True,
                    mode='markers',
                    hovertext=labels_num,
                    marker=dict(
                        size=5,
                        cmax=1,
                        cmin=0,
                        color=confidence,
                        colorbar=dict(
                            title="Confidence"
                        ),
                        colorscale=['red', 'yellow', 'green']
                    ),
                )])
        elif dropdown=='Unlabelled': 
            I = (labels=='') | (labels=='None')
            fig = go.FigureWidget(
                data=[
                go.Scatter3d(
                    x=pts[Time_Slider,I,0],
                    y=pts[Time_Slider,I,1],
                    z=pts[Time_Slider,I,2],
                    showlegend=False,
                    visible=True,
                    mode='markers',
                    hovertext=labels_num[I], 
                    marker=dict(
                        size=5,
                        cmax=1,
                        cmin=0,
                        color=confidence[I],
                        colorbar=dict(title="Confidence"),
                        colorscale=['red', 'yellow', 'green']
                    ),
                )])
            fig.add_trace(go.Scatter3d(
                x=pts[Time_Slider,~I,0],
                y=pts[Time_Slider,~I,1],
                z=pts[Time_Slider,~I,2],
                mode='markers',
                    hovertext=labels_num[~I],
                    showlegend=False,
                    marker=dict(
                        size=2,
                        color="DarkSlateGrey",),
                )
            )
        elif dropdown=='Segments':
            
            segclr = -1 * np.ones(pts.shape[1])
            for i in range(labels.shape[0]):
                if labels[i] in markers:
                    segclr[i] = segID[markers.index(labels[i])]
                else:
                    segclr[i] = np.nan
            I = np.isnan(segclr)
            fig = go.FigureWidget(
                data=[
                go.Scatter3d(
                    x=pts[Time_Slider,I,0],
                    y=pts[Time_Slider,I,1],
                    z=pts[Time_Slider,I,2],
                    mode='markers',
                    visible=True,
                    hovertext=labels_num[I],
                    showlegend=False,
                    marker=dict(
                        size=2,
                        color="DarkSlateGrey",),
                )])
            for i in range(len(uniqueSegs)):
                fig.add_trace(go.Scatter3d(
                    x=pts[Time_Slider,segclr==i,0],
                    y=pts[Time_Slider,segclr==i,1],
                    z=pts[Time_Slider,segclr==i,2],
                    mode='markers',
                    hovertext=labels_num[segclr==i],
                    showlegend=True,
                    name=uniqueSegs[i],
                    marker=dict(
                        size=5,
                        cmax=np.nanmax(segclr),
                        cmin=0,
                        color=i,
                        colorscale='hsv'),
                ))          
        
        # Center & Scale Axis on Graph
        fig.update_layout(
            autosize=False,
            scene = dict(
                xaxis = dict(nticks=10, range=[mid_x - max_range,mid_x + max_range],),
                yaxis = dict(nticks=10, range=[mid_y - max_range, mid_y + max_range],),
                zaxis = dict(nticks=10, range=[mid_z - max_range, mid_z + max_range],),
                aspectmode="cube"
            ),
            hoverlabel=dict(
                bgcolor="white", 
                font_size=11, 
                font_family="Georgia"
            ),
            uirevision='same', # don't reset camera zoom and rotation on update
            margin=dict(r=250, l=0, b=10, t=0))
           
    return fig;
    
#Export to c3d
@app.callback(Output('export_comment', 'children'), 
              [Input('export_button', 'n_clicks'), Input('pts_current', 'children'), 
               Input('labels_current', 'children'), Input('confidence_current', 'children'),
               Input('frame_rate', 'children'),Input('rotang','children')], 
              [State('upload_data', 'value')])

def export_c3d(n_clicks, pts, labels, confidence, fs, rotang, filename):
    
    export_comment=''
    
    if filename is not None:
        output_name = filename[:-4]+'_labelled.c3d'
        
        if n_clicks is None:
            raise PreventUpdate
        elif n_clicks is not None:        
            print('exporting to ' + filename)    
            pts=np.array(pts).astype('float64') 
            #rotang = float(rotang) * np.pi/180 
            aml.export_labelled_c3d(pts,labels,rotang,filename,output_name,markers,gapfillsize)

            export_comment="exported " + output_name.split(os.path.sep)[-1]
  
    return export_comment
    
print("Opening Web Browser")
webbrowser.open('http://127.0.0.1:8050/', new=2)

if __name__ == '__main__':
    app.run_server(debug=False) 