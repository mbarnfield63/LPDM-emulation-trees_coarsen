import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
import trees_emulator.marco_functions

"""
Reproducible code for "A machine learning emulator for Lagrangian particle dispersion model footprints: A case study using NAME" 
by Elena Fillola, Raul Santos-Rodriguez, Alistair Manning, Simon O'Doherty and Matt Rigby (2022)

Author: Elena Fillola (elena.fillolamayoral@bristol.ac.uk)
"""

def train_tree(data, inputs, frequency, hours_back, tree):
    """
    Train a GBRT (ie the regressor for a single cell) using parameters provided following method from paper. 
    Returns a trained sklearn regressor
    
    Requires inputs:
    - data (LoadData object)
    - inputs (meteorological inputs)
    - frequency (freq to sample footprints)
    - hours_back (same parameter as used to produce inputs)
    - tree (int ID of cell to train, if all cells are flattened. tree is then unraveled into x,y coordinates within the domain, eg tree 0 is at loc 0,0)
    
    Parallel predicting function is predict_tree. Edit/replace both of these for custom training/predicting 
    """

    center = trees_emulator.marco_functions.find_center_indexes(data.size, data.coarsen_factor)
    if tree in center:
        clf = DummyRegressor(strategy='constant', constant=0)
    else:
        sizediff = int((data.metsize-data.size)/2)
        centre = int(data.size/2)

        (x,y) = np.unravel_index(tree, (data.size,data.size))
        xtree, ytree = x + sizediff,  y + sizediff
        around_tree_indeces = []
        for i in [-2,0,+2]:
            for j in [-2,0,+2]:
                around_tree_indeces.append((xtree+i, ytree+j))

        # fixed indices (same for every tree)
        around_measurement_indeces = [(centre+sizediff,j+centre+sizediff) for j in [0,-1,1]] + [(j + centre+sizediff, centre+sizediff) for j in [-1,1]] + [(i+centre+sizediff, j+centre+sizediff) for (i,j) in [(-1,-1), (+1,+1), (+1,-1), (-1,+1)]]

        indeces = around_measurement_indeces + around_tree_indeces

        ## find flattened indeces from 2D coordinates for a single input (ie indices to keep in a metsize x metsize grid)
        idx_list_tree =  [np.ravel_multi_index([x,y], (data.metsize, data.metsize)) for (x,y) in indeces]

        ## repeat indeces for each of the concatenated inputs
        idxs = [i + (data.metsize**2)*n for n in range(int(np.shape(inputs)[1]/(data.metsize**2))) for i in idx_list_tree]

        ## select inputs
        inputs_here = inputs[:, idxs]

        ## select inputs/outputs by frequency
        X_reduced = inputs_here[::frequency,:]
        y_reduced = data.fp_data[hours_back:-3:frequency, tree]
        
        clf = GradientBoostingRegressor(n_estimators = 150, max_depth = 50, max_features="sqrt", loss='absolute_error')
        
        clf.fit(np.float64(X_reduced), np.float64(y_reduced))    

    return clf
