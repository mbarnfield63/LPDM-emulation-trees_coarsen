from trees_emulator.load_data import *
from trees_emulator.training import *

import numpy as np
import pickle
from sklearn.dummy import DummyRegressor

import argparse

import warnings
warnings.filterwarnings('ignore')

"""
Reproducible code for "A machine learning emulator for Lagrangian particle dispersion model footprints: A case study using NAME" 
by Elena Fillola, Raul Santos-Rodriguez, Alistair Manning, Simon O'Doherty and Matt Rigby (2022)

Author: Elena Fillola (elena.fillolamayoral@bristol.ac.uk)
"""

parser = argparse.ArgumentParser(prog = "tree_emulator", description='Train footprint emulator')
parser.add_argument('site', type=str,  help='Site to train on, as string (eg "MHD")')
parser.add_argument('year', help='Time period to train on. Can be int (2016) or str ("201[4-5]")')
parser.add_argument('--freq', type=int, nargs='?', default=2, help='Sampling frequency for training data as int (eg freq of 2 means one every 2 fps will be used in training)')
parser.add_argument('--size', type=int, nargs='?', default=10, help='Size of domain to be predicted as int. Default is 10')
parser.add_argument('--hours_back', nargs='*', default=[6], help='Hours back for inputs that are passed at present and in past as int. Default is 6')
parser.add_argument('--input_size', type=int, default=6, help='size for get_all_inputs')
parser.add_argument('--coarsen', type=int, default=1, help='coarsen the resolution by a factor')
parser.add_argument('--save_name', type=str, default=None, help='name to save model')

parser.add_argument('--siteheight', nargs='?',default=None,  help='Footprints height')
parser.add_argument('--metheight', nargs='?', default=None,  help='Meteorology height')
parser.add_argument('--save_dir', nargs='?', default=None,  help='Folder to store saved model in')
parser.add_argument('--met_datadir', nargs='?', default=None,  help='Met folder')
parser.add_argument('--extramet_datadir', nargs='?', default=None,  help='Gradients folder')
parser.add_argument('--fp_datadir', nargs='?', default=None,  help='Footprints folder')

parser.add_argument('--verbose', nargs='?', type=bool, default=True, help='Print update messages')


args = parser.parse_args()

try: 
    year = int(args.year)
except:
    year = args.year
if args.verbose: print("Loading data")

if args.save_name is None:
    save_name=args.site
else:
    save_name=args.save_name
if args.save_dir==None:
    save_dir = f"/user/work/eh19374/LPDM-emulation-trees_TESTING/trained_models/{save_name}.txt"
else:
    save_dir=args.save_dir

heights = {"MHD":"10magl", "THD":"10magl", "TAC":"185magl", "RGL":"90magl", "HFD":"100magl", "BSD":"250magl", "GSN":"10magl"}
met_datadir = "/group/chemistry/acrg/met_archive/NAME/EUROPE_met/EUROPE_Met_10magl_*"
fp_datadir = f"/group/chemistry/acrg/LPDM/fp_NAME/EUROPE/{args.site}-{heights[args.site]}_UKV_EUROPE_*"
extramet_datadir = "/group/chemistry/acrg/met_archive/NAME/full_extra_vars/EUROPE*" 

# Load data
data = LoadData(year, 
                site=args.site, 
                siteheight=args.siteheight, 
                coarsen_factor=args.coarsen, 
                metheight=args.metheight, 
                size=args.size, 
                met_datadir=met_datadir, 
                extramet_datadir=extramet_datadir, 
                fp_datadir=fp_datadir, 
                verbose=args.verbose)

if args.verbose: print("Data loaded successfully")

# Set up inputs variables
## variables that are passed at the time of the footprint and x hours before
vars_with_past = [data.y_wind, data.x_wind, data.met.PBLH.values]
## variables that are only passed at the time of the footprint
vars_without_past = [data.temp_grad, data.x_wind_grad, data.y_wind_grad]
## hours back to ints
hours_back_ints = [int(hour) for hour in args.hours_back]

inputs = get_all_inputs_v3(vars_with_past, hours_back_ints, vars_without_past, size=args.input_size)

clfs = []
if args.verbose: print("Starting training")

for tree in range(args.size**2):
    clf = train_tree(data, inputs, args.freq, max(hours_back_ints), tree)
    clfs.append(clf)
    #if args.verbose: print(f"Tree {str(tree)} trained")

info = {'site': data.site, 'training data': year, 'sampling frequency': args.freq, 'size': args.size}

with open(save_dir, 'wb') as f:
    pickle.dump([info, clfs], f)

