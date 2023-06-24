import random
import logging
import numpy as np
import pandas as pd

import torch 
from torch_geometric.loader import DataLoader

from ..objects import Bering_Graph as BrGraph
from ._graph import BuildGraph
from ._settings import GRAPH_KEYS as G_KEYS

logger = logging.getLogger(__name__)

def BuildWindowGraphs(
    bg: BrGraph, 
    n_cells_perClass: int = 10, 
    window_width: float = G_KEYS.WINDOW_WIDTH, 
    window_height: float = G_KEYS.WINDOW_HEIGHT, 
    n_neighbors: int = G_KEYS.N_NEIGHBORS, 
    min_points: int = G_KEYS.WINDOW_MIN_POINTS,
    prop_unseg: float = 0.8, 
    max_unsegmented_thresh: float = 0.4,
    cell_percentile_from_border: float = 10,
    slide_window_ratio: float = 0.25,
    n_windows_per_cell: int = 5,
    min_spots_outside: int = 5,
    **kwargs,
):
    '''
    Build Graphs for originally segemented cells (labelled / golden truth)
    '''
    # init
    Spots = bg.spots_all.copy()
    Graphs = []
    bg.window_size = window_width

    # select cells and find attributes
    selected_cells = []
    cell_meta = bg.segmented.copy()
    labels = np.setdiff1d(cell_meta.labels.unique(), ['background'])

    cx_min, cx_max = np.percentile(cell_meta['cx'].values, cell_percentile_from_border), np.percentile(cell_meta['cx'].values, 100 - cell_percentile_from_border)
    cy_min, cy_max = np.percentile(cell_meta['cy'].values, cell_percentile_from_border), np.percentile(cell_meta['cy'].values, 100 - cell_percentile_from_border)

    cell_meta = cell_meta.loc[(cell_meta['cx'] > cx_min) & (cell_meta['cx'] < cx_max) & (cell_meta['cy'] > cy_min) & (cell_meta['cy'] < cy_max), :].copy() # remove cells in border
    
    for label in labels:
        cells = cell_meta.loc[cell_meta['labels'] == label, :].index.values
        selected_cells += random.sample(list(cells), min(n_cells_perClass, len(cells)))

    counts = 0
    for cell_idx, cell in enumerate(selected_cells):
        cx, cy, d = bg.segmented.loc[cell, 'cx'], bg.segmented.loc[cell, 'cy'], bg.segmented.loc[cell, 'd']

        if n_windows_per_cell == 1:
            xc_list = [cx]
            yc_list = [cy]
            # xc_list = [cx-d*slide_window_ratio]
            # yc_list = [cy+d*slide_window_ratio]
        elif n_windows_per_cell == 3:
            xc_list = [cx-d*slide_window_ratio, cx, cx+d*slide_window_ratio]
            yc_list = [cy+d*slide_window_ratio, cy, cy-d*slide_window_ratio]
        elif n_windows_per_cell == 5:
            xc_list = [cx-d*slide_window_ratio, cx-d*slide_window_ratio, cx, cx+d*slide_window_ratio, cx+d*slide_window_ratio]
            yc_list = [cy+d*slide_window_ratio, cy-d*slide_window_ratio, cy, cy+d*slide_window_ratio, cy-d*slide_window_ratio]
        
        positions = ['topleft', 'bottomleft', 'centroid', 'topright', 'bottomright']
        for xc, yc, pos in zip(xc_list, yc_list, positions):
            # define the core window
            xmin, ymin = xc - window_width / 2, yc - window_height / 2
            xmax, ymax = xc + window_width / 2, yc + window_height / 2

            window_spots = Spots.loc[(Spots.x > xmin) & (Spots.x < xmax) & (Spots.y > ymin) & (Spots.y < ymax), :].copy()
            if window_spots.shape[0] == 0:
                continue
            
            spots_abun = pd.DataFrame(window_spots.groupby(['segmented']).size(), columns = ['counts'])
            spots_abun.sort_values(by = ['counts'], ascending = False, inplace = True)
            if spots_abun.shape[0] == 1 or spots_abun.iloc[1,0] < min_spots_outside:
                continue

            window_seg = window_spots.loc[window_spots['groups'] == 'segmented', :].copy()
            window_unseg = window_spots.loc[window_spots['groups'] == 'unsegmented', :].copy()
            ratio_unseg = window_unseg.shape[0] / (window_unseg.shape[0] + window_seg.shape[0])

            if ratio_unseg >= max_unsegmented_thresh:
                continue

            # get subset of unlabelled spots
            unseg_indices = random.sample(list(range(window_unseg.shape[0])), int(window_unseg.shape[0] * prop_unseg))
            window_unseg = window_unseg.iloc[unseg_indices,:].copy()
            window_spots = pd.concat([window_seg, window_unseg], axis = 0)

            if (window_spots.shape[0] < n_neighbors + 1) or (window_spots.shape[0] < min_points):
                continue
            
            counts += 1
            graph = BuildGraph(bg, window_spots, n_neighbors = n_neighbors, **kwargs)
            Graphs.append(graph)

            if counts % 50 == 0:
                logger.info(f'Build Neighbor graphs for {counts} th window (golden truth)')
                logger.info(f'Number of dots in {counts} th window: {window_spots.shape[0]}')
                
                avg_neighbors = graph.edge_index.shape[1] / graph.x.shape[0]
                logger.info(f'Average number of filtered neighbors: {avg_neighbors:.2f} in the window')
    
    bg.Graphs_golden = Graphs
    logger.info(f'Number of node features: {bg.n_node_features}')
    logger.info(f'\nTotal number of golden-truth graphs is {len(bg.Graphs_golden)}')

def CreateData(
    bg: BrGraph, 
    batch_size: int = G_KEYS.BATCH_SIZE, 
    training_ratio: float = G_KEYS.TRAINING_RATIO
):
    '''
    Create training and testing data loader
    '''
    # initialize
    logger.info(f'Create training and testing datasets (golden truth)')
    dataset = bg.Graphs_golden

    N = len(dataset)
    N_train = int(N * training_ratio)
    
    random.shuffle(dataset)

    train_loader = DataLoader(dataset[:N_train], batch_size = batch_size, shuffle = False, num_workers = 0)
    test_loader = DataLoader(dataset[N_train:], batch_size = batch_size, shuffle = False, num_workers = 0)
    
    bg.train_loader = train_loader
    bg.test_loader = test_loader

    # bg.Graphs_train = dataset[:N_train]
    # bg.Graphs_test = dataset[N_train:]
    del bg.Graphs_golden