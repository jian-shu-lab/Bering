import os
import random
import torch
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Sequence

import leidenalg as la
from sklearn.metrics import adjusted_rand_score

import matplotlib as mpl
import matplotlib.pyplot as plt

from torch_geometric.data import Data 

from ._settings import _PLOT_SETTINGS, _GET_CMAPS
from ._settings import PLOT_KEYS as PLT_KEYS

from ..graphs import BuildGraph, BuildGraph_fromRaw
from ..segment import find_clusters_predictedLinks
from ..objects import Bering_Graph as BrGraph
from ._plot_elements import _raw_spots, _raw_cell_types, _raw_cell_types_addPatch
from ._plot_elements import _predicted_cell_types, _predicted_probability, _draw_cells_withStaining

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_PLOT_SETTINGS()
CMAP = _GET_CMAPS()

def _get_extended_window(spots, location, window_width = 50.0, window_height = 50.0):
    loc_x, loc_y = location
    minx, miny = loc_x - window_width, loc_y - window_height
    maxx, maxy = loc_x + window_width, loc_y + window_height
    logger.info(f'loc_x, loc_y: {loc_x}, {loc_y}')
    logger.info(f'minx, miny, maxx, maxy: {minx}, {miny}, {maxx}, {maxy}')

    window_spots = spots.loc[(spots['x'] > minx) & (spots['x'] < maxx) & (spots['y'] > miny) & (spots['y'] < maxy), :].copy()
    return window_spots

def _get_graph(bg, location, Spots, n_neighbors = 10, window_size = 50.0):
    # get spots in window
    window_spots = _get_extended_window(Spots.copy(), location, window_size, window_size)
    # logger.info(f'window size for location {location} is {window_size.shape}')

    # build neighbor graph
    try:
        graph = BuildGraph_fromRaw(bg, window_spots, bg.features.copy(), n_neighbors = n_neighbors).cpu()
    except AssertionError:
        raise Exception('No enough transcripts in this window')

    return window_spots, graph

def _prediction_nodes(graph, trainer_node, n_labels, prob_threshold = 0.3):
    preds_q = trainer_node.predict(graph, device = 'cpu').cpu()
    max_probs, predictions = torch.max(preds_q, dim = 1)

    back_indices = torch.where(max_probs <= prob_threshold)[0].unsqueeze(1)
    predictions[back_indices] = n_labels - 1

    return predictions, max_probs

def Plot_Classification_Post(
    bg: BrGraph, 
    location: Sequence[float], 
    n_neighbors: int = 10, 
    min_prob: float = 0.3, 
    window_size: float = 50.0,
):
    '''
    Plot original spots and newly-segmented spots
    '''
    # BUILD GRAPHS
    if len(bg.label_to_col) == 0:
        bg.label_to_col = dict(zip(bg.labels, CMAP[:bg.n_labels]))
        bg.label_to_col['background'] = '#C0C0C0'

    Spots = bg.spots_all.copy()

    df_window_raw, graph = _get_graph(
        bg, location, Spots, n_neighbors = n_neighbors, window_size = window_size,
    )
    graph = graph.cpu()
    predictions, max_probs = _prediction_nodes(
        graph, bg.trainer_node, bg.n_labels, prob_threshold = min_prob,
    )
    predictions = predictions.numpy()
    max_probs = max_probs.numpy()

    # PREPARATION
    tps = graph.pos[:, 0].cpu().numpy()
    x = graph.pos[:, 1].cpu().numpy()
    y = graph.pos[:, 2].cpu().numpy()

    pred_labels = np.array([bg.label_indices_dict[i] for i in predictions])
    label_to_col = bg.label_to_col

    # PLOTS
    fig, axes = plt.subplots(figsize = (PLT_KEYS.AX_WIDTH * 3, PLT_KEYS.AX_HEIGHT * 1), ncols = 3, nrows = 1, sharex = True, sharey = True) 
    if bg.image_raw is not None:
        dapi = bg.image_raw[0,:,:]
    else:
        dapi = None

    axes[0] = _raw_spots(dapi, x, y, axes[0])
    axes[1] = _predicted_cell_types(dapi, x, y, axes[1], pred_labels, label_to_col)
    axes[1].set_title(f'Predicted Annotations')
    axes[2], prob_plot = _predicted_probability(dapi, x, y, axes[2], max_probs)
    plt.colorbar(prob_plot, ax = axes[2])
    fig.tight_layout()

    # SAVE
    output_name = f'{PLT_KEYS.FOLDER_NAME}/{PLT_KEYS.FILE_PREFIX_CLASSIFICATION}x_{(location[0]):.2f}_y_{(location[1]):.2f}{PLT_KEYS.FILE_FORMAT}'
    fig.savefig(output_name, bbox_inches = 'tight')

    # FOREGROUND WINDOW
    df_window_pred = df_window_raw.iloc[np.where(pred_labels != 'background')[0],:].copy()
    return df_window_raw, df_window_pred, predictions

def Plot_Segmentation_Post(
    bg: BrGraph, 
    location: Sequence[float],
    df_window_raw: pd.DataFrame = None, 
    df_window_pred: pd.DataFrame = None,
    predictions: np.ndarray = None,
    n_neighbors: int = 10, 
    window_size: float = 50.0,
    use_image: bool = True,
    pos_thresh: float = 0.6,
    resolution: float = 0.05,
    num_edges_perSpot: int = 300,
    min_prob_nodeclf: float = 0.3,
    n_iters: int = 10,
):
    '''
    Plot Original Cell IDs and Cell ID distribution on latent space
    Either input a cell name and then extract a table or a pre-filtered spots window.
    '''
    Spots = bg.spots_all.copy()

    if df_window_raw is None:
        # clean background nodes
        df_window_raw, graph = _get_graph(
            bg, location, Spots, n_neighbors = n_neighbors, window_size = window_size,
        )
        graph = graph.cpu()
        predictions, max_probs = _prediction_nodes(graph, bg.trainer_node, bg.n_labels, prob_threshold = min_prob_nodeclf)
        predictions = predictions.numpy()
        max_probs = max_probs.numpy()
        
        df_window_pred = df_window_raw.iloc[np.where(pred_labels != 'background')[0], :].copy()
        
    pred_labels = np.array([bg.label_indices_dict[i] for i in predictions])
    
    cells_pred = np.zeros(df_window_raw.shape[0], dtype = int)

    if df_window_pred.shape[0] < num_edges_perSpot:
        return

    cls_predlink = find_clusters_predictedLinks(
        bg, 
        df_spots = df_window_pred, # foreground only
        use_image = use_image,
        pos_thresh = pos_thresh,
        resolution = resolution,
        num_edges_perSpot = num_edges_perSpot,
        n_neighbors = n_neighbors,
        num_iters = n_iters,
    )[resolution]
    cells_pred[np.where(pred_labels != 'background')[0]] = cls_predlink
    cells_pred[np.where(pred_labels == 'background')[0]] = -1

    x_raw, y_raw = df_window_raw.x.values, df_window_raw.y.values
    x_predfore, y_predfore = df_window_pred.x.values, df_window_pred.y.values

    if bg.channels is not None:
        fig, axes = plt.subplots(figsize = (PLT_KEYS.AX_WIDTH * (bg.n_channels + 1) * 0.8, PLT_KEYS.AX_HEIGHT * 0.8), ncols = bg.n_channels + 1, nrows = 1, sharex = True, sharey = True)
        for idx in range(bg.n_channels):
            axes[idx] = _raw_spots(bg.image_raw[idx], x_raw, y_raw, axes[idx], 'Raw (w. ' + bg.channels[idx] + ')')

        axes[-1] = _draw_cells_withStaining(bg.image_raw[0], x_predfore, y_predfore, axes[-1], cls_predlink, 'Pred (w. DAPI)')

    else:
        fig, axes = plt.subplots(figsize = (PLT_KEYS.AX_WIDTH * 2 * 0.8, PLT_KEYS.AX_HEIGHT * 0.8), ncols = 2, nrows = 1, sharex = True, sharey = True)
        axes[0] = _raw_spots(None, x_raw, y_raw, axes[0])
        axes[1] = _draw_cells_withStaining(None, x_predfore, y_predfore, axes[1], cls_predlink, 'Pred Cells')

    fig.tight_layout()

    output_name = f'{PLT_KEYS.FOLDER_NAME}/{PLT_KEYS.FILE_PREFIX_SEGMENTATION}x_{(location[0]):.2f}_y_{(location[1]):.2f}{PLT_KEYS.FILE_FORMAT}'
    fig.savefig(output_name, bbox_inches = 'tight')
    plt.close(fig)