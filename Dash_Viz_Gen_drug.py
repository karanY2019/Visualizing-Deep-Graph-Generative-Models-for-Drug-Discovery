# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 09:36:10 2020

@author: Karan

python 0822zinc.py
"""

from time import sleep
import dash_bootstrap_components as dbc
from random import randint, seed

import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

import dash_table

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_bio_utils import xyz_reader
from molmod import *
import dash_bio
import dash_bio as dashbio
import pandas as pd
import plotly.graph_objs as go
import os
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Draw
import base64
from io import BytesIO
import io
from PIL import Image, ImageFile
import json
import utils.environment as env
import numpy as np
###load data model


from optimize_property import MoFlowProp
from optimi_dash0819 import *

###zinc

zinc250_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]  # 0 is for virtual node.
max_atoms = 38
n_bonds = 4
data_name = 'zinc250k'
atomic_num_list = zinc250_atomic_num_list
        # transform_fn = transform_qm9.transform_fn
transform_fn = transform_zinc250k.transform_fn_zinc250k
        # true_data = TransformDataset(true_data, transform_fn_zinc250k)
valid_idx = transform_zinc250k.get_val_ids()
molecule_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
temperature=0.65
delta=0.5
data_dir='data'
model_dir='results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask'
gen_dir = os.path.join(model_dir, 'generated')
snapshot_path='model_snapshot_epoch_200'
snapshot_path = os.path.join(model_dir, snapshot_path)
hyperparams_path='moflow-params.json'
hyperparams_path = os.path.join(model_dir, hyperparams_path)

dataset = NumpyTupleDataset(os.path.join(data_dir, molecule_file), transform=transform_fn)  # 133885

#dataset=np.load('zincdataset.npy',allow_pickle=True)
   
assert len(valid_idx) > 0
train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
n_train = len(train_idx)  # 120803
    # train_idx.extend(valid_idx) # 120803 + last 13082 for validation = 133885 intotal
    # train, test = chainer.datasets.split_dataset(dataset, n_train, train_idx)
train = torch.utils.data.Subset(dataset, train_idx)  # 120803
test = torch.utils.data.Subset(dataset, valid_idx)  # 13082  not used for ge

##model
model_params = Hyperparameters(path=hyperparams_path)
model = MoFlow(model_params) 
model = load_model(snapshot_path, model_params, debug=True)
device = torch.device('cpu')    
#gpu = 0
        # device = args.gpu
#device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
#model=model.cuda()    
model.to(device)
model.eval()  # Set model for 

##opt model
import time
start = time.time()
property_model_path='plogp_model.pt'
#property_model_path='qed_model.pt'
print("Load regression model and do optimization")
prop_list = load_property_csv(data_name, normalize=False)
train_prop = [prop_list[i] for i in train_idx]
test_prop = [prop_list[i] for i in valid_idx]
print('Prepare data done! Time {:.2f} seconds'.format(time.time() - start))
property_model_path = os.path.join(model_dir, property_model_path)
        #property_model_path = os.path.join(args.model_dir,'qed_discovered_sorted_bytop2k.csv')
print("loading {} regression model from: {}".format( data_name,property_model_path))

hidden =""
if hidden in ('', ','):
        hidden = []
else:
        hidden = [int(d) for d in hidden.strip(',').split(',')]
print('Hidden dim for output regression: ', hidden)

#property_model = MoFlowProp(model, hidden)   
property_model = torch.load(property_model_path, map_location=device)
print('Load model done! Time {:.2f} seconds'.format(time.time() - start))
property_model.to(device)
property_model.eval()
###############**************************************#############################

##speck data
#mol_optm = Molecule.from_file("dash_output/constr_opt_input.sdf")
#mol_optm.write_to_file("dash_output/constr_opt_input.xyz")
#data=xyz_reader.read_xyz('dash_output/constr_opt_input.xyz')

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])#https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/cerulean/bootstrap.min.css
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally=True

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "1rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Viz MoFlow", className="display-4"),
        html.Hr(),
        html.P(
            "Visualization tasks: ", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("random generation", href="/page-1", id="page-1-link"),
                dbc.NavLink("Explore Latent Space", href="/page-2", id="page-2-link"),
                dbc.NavLink("Optimization I", href="/page-3", id="page-3-link"),
                dbc.NavLink("Optimization II", href="/page-4", id="page-4-link"),
            ],
            vertical=True,
            pills=True,
        ),
        html.H4("select Model"),
        dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'MoFlow', 'value': 'MolFlow'},
            {'label': 'graphvpn', 'value': 'graphvpn'},
            {'label': 'VAE', 'value': 'VAE'}
        ],
        value='MoFlow'
    ),
        html.H4("select Dataset"),
        dcc.Dropdown(
        id='data-dropdown',
        options=[
            {'label': 'zinc250k', 'value': 'zinc250k'},
            {'label': 'qm9', 'value': 'qm9'},
            {'label': 'other', 'value': 'other'}
        ],
        value='zinc250k'),
        
        
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

tabs = html.Div([dcc.Location(id="url"), sidebar, content])


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 5)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return False, False, False,False
    return [pathname == f"/page-{i}" for i in range(1, 5)]


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]: #   return html.P("This is the content of page 1!"),d
        return html.P("This is the content of page 1!"),random_generation,d
   
    elif pathname == "/page-2":
        return interpolation
    #html.P("This is the content of page 2. Yay!")
    
    elif pathname == "/page-3":
        return html.P("Oh cool, this is page 3!"),vizopt
    
    elif pathname == "/page-4":
        return html.P("Oh cool, this is page 4!"),vizopt_cstr
   
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )
###3D slider    
default_sliders = [
    html.Div(className='app-controls-block', children=[
        html.Div(
            "Atom radius",
            className='app-controls-name'
        ),
        dcc.Slider(
            id='speck-atom-radius',
            className='control-slider',
            max=1,
            step=0.01,
            value=0.6,
            updatemode='drag'
        ),
    ]),
    html.Div(className='app-controls-block', children=[
        html.Div(
            "Relative atom radius",
            className='app-controls-name'
        ),
        dcc.Slider(
            id='speck-relative-atom-radius',
            className='control-slider',
            max=1,
            step=0.01,
            value=1.0,
            updatemode='drag'
        ),
    ]),
    html.Div(className='app-controls-block', children=[
        html.Div(
            "Ambient occlusion",
            className='app-controls-name'
        ),
        dcc.Slider(
            id='speck-ao',
            className='control-slider',
            max=1,
            step=0.01,
            value=0.75
        ),
    ]),
    html.Div(className='app-controls-block', children=[
        html.Div(
            "Brightness",
            className='app-controls-name'
        ),
        dcc.Slider(
            id='speck-brightness',
            className='control-slider',
            max=1,
            step=0.01,
            value=0.5,
            updatemode='drag'
        )
    ]),
    html.Div(className='app-controls-block', children=[
        html.Div(
            "Outline",
            className='app-controls-name'
        ),
        dcc.Slider(
            id='speck-outline',
            className='control-slider',
            max=1,
            step=0.01,
            value=0.0,
            updatemode='drag'
        ),
    ]),
    #html.Hr(),
    dcc.Checklist(
        id='speck-show-hide-bonds',
        options=[
            {'label': 'Show bonds',
             'value': 'True'}
        ],
        value=[]
    ),
    html.Div(className='app-controls-block', children=[
        html.Div(
            'Bond scale',
            className='app-controls-name'
        ),
        dcc.Slider(
            id='speck-bond-scale',
            className='control-slider',
            max=1,
            step=0.01,
            value=0.5,
            updatemode='drag'
        )
    ])

]
    
#######PAGE2
controls_page2 = dbc.Card(
    [   
        dbc.FormGroup(               
            [                     
                dbc.Label("Enter SMILES seed0 and seed1: "), 
                html.Div(
    [
            dbc.Input(id="seed0-input", placeholder="Type seed0...", type="text",size=60),
            html.Hr(),
            dbc.Input(id="seed1-input", placeholder="Type send1...", type="text",size=60),
            html.Hr(),          
            dbc.Button("View Linear Interpolation", id="button-linearIntpol",color="info",className="mr-1"),
            
           ])])]
   ,body=True,),
    
interpolation = dbc.Container(
    [
        html.H1(" Explore Molecular Latent Space",style={'backgroundColor':'#DCDCDC'}), 
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls_page2, md=8),
                              
            ]),
         dbc.Row(
            [                
                dbc.Col(html.Img(id="linearIntpol-img"), md=12)                
            ]),
    ]),#align="center",),
   
    ###***run model function###***
@app.callback(    
    Output('linearIntpol-img', 'src'),
    [Input("button-linearIntpol", "n_clicks")],
    [State('seed0-input', 'value'),
    State('seed1-input', 'value')],
    )
def update_output_linearIntpol(n_clicks,s0,s1):
    smile0=str(s0)
    smile1=str(s1)
    #filepath = os.path.join(gen_dir, '2points_interpolation_molecules')
    src=visualize_interpolation_between_2_points_input(smile0,smile1, model, mols_per_row=10, n_interpolation=100,
                                    atomic_num_list=atomic_num_list, seed=1, true_data=train,
                                    device=device,data_name=data_name)
    return src

    
                
    
controls_page4 = dbc.Card(
    [   
        dbc.FormGroup(               
            [  
                    
                dbc.Label("Constrained by : "),
        dbc.Checklist(
                 options=[
               {'label': 'Tanimoto similarity', 'value': 'tsim'},
               {'label': 'option 2', 'value': 'qed'},
               {'label': 'option 3', 'value': 'plogp'}
                          ],
               value=['tsim'],
               labelStyle={'display': 'inline-block'}),
                
                dbc.Label('choose a property:'),
                dcc.Dropdown(
                    id="image-dropdown",
                    options=[
                {'label': 'QED', 'value': 'QED'},
                {'label': 'Plogp', 'value': 'Plogp'}
               
            ],
               
            value='Plogp',
                ),
            #html.Hr(),   
            dbc.Button("Optimize", id="button",color="info", className="mr-1"),
            ]
        ),
        
    ],
    body=True,
)
                
vizopt_cstr = dbc.Container(
    [
        html.H1(" Optimization",style={'backgroundColor':'#DCDCDC'}), 
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Input(id="input", placeholder="Type something...", type="text",size=60),)]),
                
        dbc.Row(
            [
                dbc.Col(controls_page4, md=3),
                dbc.Col(html.Img(id="output-img"), md=6)
                
            ],
            #align="center",
        ),
        
        html.P(id='output-smile'),        
                      #html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.Div(id='table'),)
            ],),#)align="center",),
        
        #html.Hr(),              
        dbc.Row([dbc.Col(html.Div(id='speck-control-tabs', className='control-tabs', 
                                  children=[dcc.Tabs(id='speck-tabs', value='what-is', children=[

                dcc.Tab(
                    label='About',
                    value='what-is',
                    children=html.Div(className='control-tab', children=[
                        html.H4(className='what-is', children='What is Speck?'),
                        html.P('Speck is a WebGL-based molecule renderer. By '
                               'using ambient occlusion, the resolution of '

                               'the rendering does not suffer as you zoom in.'),
                        html.P('You can toggle between molecules using the menu under the '
                               '"Data" tab, and control parameters related to '
                               'the appearance of the molecule in the "View" tab. '
                               'These parameters can be controlled at a low level '
                               'with the sliders provided, or preset views can be '
                               'applied for a higher-level demonstration of changing '
                               'atom styles and rendering.')
                        ])
                ),
                
            dcc.Tab(
                    label='View',
                    value='view-options',
                    children=html.Div(className='control-tab', children=[
                        dcc.Checklist(
                            id='speck-enable-presets',
                            options=[{'label': 'Use presets', 'value': 'True'}],
                            value=[]
                        ),
                        #html.Hr(),
                        html.Div(id='speck-controls-detailed', children=default_sliders),
                        html.Div(
                            id='speck-controls-preset',
                            className='speck-controls',
                            children=[
                               html.Div(className='app-controls-block', children=[
                                    html.Div(className='app-controls-name',
                                             children='Rendering style'),
                                    dcc.Dropdown(
                                        id='speck-preset-rendering-dropdown',
                                        className='speck-dropdown',
                                        options=[
                                            {'label': 'Default/reset',
                                             'value': 'default'},
                                            {'label': 'Toon',
                                             'value': 'toon'},
                                        ],
                                        value='default'
                                    )
                                ]),
                                html.Div(className='app-controls-block', children=[
                                    html.Div(className='app-controls-name', children='Atom style'),
                                    dcc.Dropdown(
                                        id='speck-preset-atom-style-dropdown',
                                        className='speck-dropdown',
                                        options=[
                                            {'label': 'Ball-and-stick',
                                             'value': 'stickball'},
                                            {'label': 'Licorice',
                                             'value': 'licorice'}
                                        ],
                                        value='default'
                                    )                               ])
                            ]
                        )
                    ]),
                     
                ),  
            ]),
                                      
                dcc.Store(
            id='speck-store-preset-rendering',
            data=None
        ),

        dcc.Store(
            id='speck-store-preset-atom-style',
            data=None
        ),                                                                            
                                      ]),md=4),
                                                     
                 dbc.Col(dash_bio.Speck(
                    id='speck',
                    #data=data,
                    view={'resolution': 600, 'zoom': 0.3},
                    scrollZoom=True), md=8)
                                 
   ]), 
    ])
        
"""
        'Choose smiles to optimize: ',
    dcc.Dropdown(
        id='smiles-dropdown',
        options=[
            {'label': 'C[NH+]1CCc2ccccc2Cc2[nH]c3ccccc3c2CC1 ','value': 'C[NH+]1CCc2ccccc2Cc2[nH]c3ccccc3c2CC1'}, #ugly
            {'label': 'Nc1cc2c3c(c1)CCC[NH+]3CCC2','value': 'Nc1cc2c3c(c1)CCC[NH+]3CCC2'}, #-23 ok
            {'label': 'Cc1ccc(NC(=O)Nc2cc3c4c(c2)CCN4C(=O)CC3)cc1C ','value': 'Cc1ccc(NC(=O)Nc2cc3c4c(c2)CCN4C(=O)CC3)cc1C'}, #great
            {'label': 'Cc1ccc2c(c1)c1c3n2CC[NH+](CCC#N)[C@@H]3CCC1  ','value': 'Cc1ccc2c(c1)c1c3n2CC[NH+](CCC#N)[C@@H]3CCC1 '} #-19 good
           
        ],value='C[NH+]1CCc2ccccc2Cc2[nH]c3ccccc3c2CC1'
    ),  
"""        
@app.callback(
    #Output('output-img', 'src'),
    [Output('speck', 'data'),
    Output('output-img', 'src'),
    Output('output-smile', "children"),   
    ],
      
    #Output('output', 'children')],#],#'children'    
    [Input("button", "n_clicks")],
    [State('input', 'value')])
def update_output(n_clicks,value):
    smile=str(value)
    slist=[]
    slist.append(smile)
    
    ##mol = [Chem.MolFromSmiles(s) for s in slist]
    ##img = Draw.MolsToGridImage(mol,  molsPerRow=1,subImgSize=(250, 200),)
    ## plogp = [env.penalized_logp(m) for m in mol]          
    
    #run model optimization
    property_name='plogp'
    debug=True
    src,new_smiles=constrain_optimization_smiles_input(smile,model, property_model,device, data_name, property_name,
                                  atomic_num_list, debug, sim_cutoff=0.0) 
    mol_optm = Molecule.from_file("dash_output/constr_opt_input.sdf")
    mol_optm.write_to_file("dash_output/constr_opt_input.xyz")
    data=xyz_reader.read_xyz('dash_output/constr_opt_input.xyz')
    return data,src,"Optimized SMILES: {}".format(new_smiles)


####3D 
@app.callback(
        Output('speck', 'view'),
        [Input('speck-enable-presets', 'value'),
         Input('speck-atom-radius', 'value'),
         Input('speck-relative-atom-radius', 'value'),
         Input('speck-show-hide-bonds', 'value'),
         Input('speck-bond-scale', 'value'),
         Input('speck-ao', 'value'),
         Input('speck-brightness', 'value'),
         Input('speck-outline', 'value')]
    )
def change_view(
            presets_enabled,
            atom_radius,
            relative_atom_radius,
            show_bonds,
            bond_scale,
            ambient_occlusion,
            brightness,
            outline
    ):
        return {
            'atomScale': atom_radius,
            'relativeAtomScale': relative_atom_radius,
            'bonds': bool(len(show_bonds) > 0),
            'bondScale': bond_scale,
            'ao': ambient_occlusion,
            'brightness': brightness,
            'outline': outline
        }

@app.callback(
        Output('speck-store-preset-rendering', 'data'),
        [Input('speck-preset-rendering-dropdown', 'value')]
    )
def update_rendering_option(render):
        return render

@app.callback(
        Output('speck-store-preset-atom-style', 'data'),
        [Input('speck-preset-atom-style-dropdown', 'value')]
    )
def update_atomstyle_option(atomstyle):
        return atomstyle

@app.callback(
        Output('speck', 'presetView'),
        [Input('speck-store-preset-atom-style', 'modified_timestamp'),
         Input('speck-store-preset-rendering', 'modified_timestamp')],
        state=[State('speck-preset-rendering-dropdown', 'value'),
               State('speck-preset-atom-style-dropdown', 'value')]
    )
def preset_callback(
            atomstyle_ts, render_ts,
            render, atomstyle
    ):
        preset = 'default'
        if atomstyle_ts is None and render_ts is None:
            return preset
        if atomstyle_ts is not None and render_ts is None:
            preset = atomstyle
        elif atomstyle_ts is None and render_ts is not None:
            preset = render
        else:
            if render_ts > atomstyle_ts or atomstyle is None:
                preset = render
            else:
                preset = atomstyle
        return preset

@app.callback(
        Output('speck-preset-atom-style-dropdown', 'value'),
        [Input('speck-preset-rendering-dropdown', 'value')],
        state=[State('speck-preset-atom-style-dropdown', 'value')]
    )
def keep_atom_style(render, current):
        if render == 'default':
            return None
        return current
    
app.layout = dbc.Container(
    [tabs 
        ],
   
    #fluid=True, 
)

if __name__ == '__main__':
    app.run_server(debug=True,port=8080)
    
    
