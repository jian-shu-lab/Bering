import os
import random
import torch
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Optional

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
from ._plot_elements import _raw_segmentation, _raw_cell_types, _raw_cell_types_addPatch
from ._plot_elements import _predicted_cell_types, _predicted_probability, _draw_cells_withStaining, _draw_cells_withStaining_convexhull

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_PLOT_SETTINGS()
CMAP = _GET_CMAPS()

def _get_cell_centroid_window(spots, cell_metadata, cell_name, zoomout_scale_x = 8.0, zoomout_scale_y = 8.0):
    cx, cy = cell_metadata.loc[cell_name, 'cx'], cell_metadata.loc[cell_name, 'cy']
    dx, dy = cell_metadata.loc[cell_name, 'dx'], cell_metadata.loc[cell_name, 'dy']

    window_x_min, window_x_max = cx - zoomout_scale_x * dx, cx + zoomout_scale_x * dx
    window_y_min, window_y_max = cy - zoomout_scale_y * dy, cy + zoomout_scale_y * dy
    
    # get spots within window
    window_spots = spots.loc[(spots['x'] > window_x_min) & (spots['x'] < window_x_max) & (spots['y'] > window_y_min) & (spots['y'] < window_y_max), :].copy()
    return window_spots

def _get_extended_window(graph, spots, window_width = 50.0, window_height = 50.0):
    x = graph.pos[:,1].cpu().numpy()
    y = graph.pos[:,2].cpu().numpy()
    cx, cy = np.mean(x), np.mean(y)
    cx, cy = np.round(cx, 2), np.round(cy, 2)

    minx, miny = cx - window_width, cy - window_height
    maxx, maxy = cx + window_width, cy + window_height
    window_spots = spots.loc[(spots['x'] > minx) & (spots['x'] < maxx) & (spots['y'] > miny) & (spots['y'] < maxy), :].copy()

    return window_spots

def _get_graph(bg, cell_name, Spots, cell_metadata, n_neighbors = 10, **kwargs):
    # get spots in window
    window_spots = _get_cell_centroid_window(Spots.copy(), cell_metadata, cell_name, **kwargs)

    # build neighbor graph
    try:
        # graph = BuildGraph(bg, window_spots, n_neighbors = n_neighbors).cpu()
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

def Plot_SliceImages(
    bg: BrGraph,
):
    '''
    Plot the whole slice, with individual segmented cells as spots.

    Parameters
    ----------
    bg: BrGraph
        Bering Graph object
    '''

    if len(bg.label_to_col) == 0:
        bg.label_to_col = dict(zip(bg.labels, CMAP[:bg.n_labels]))
        bg.label_to_col['background'] = '#C0C0C0'

    df_segmented = bg.segmented.copy()
    df_labels = bg.spots_seg.copy().drop_duplicates(['segmented', 'labels'])
    df_labels.set_index('segmented', inplace = True)
    df_segmented['labels'] = df_labels.loc[df_segmented.index.values, 'labels'].values
    
    # data
    x, y = df_segmented.cx.values, df_segmented.cy.values
    celltypes = df_segmented.labels
    
    fig, ax = plt.subplots(
        figsize = (PLT_KEYS.AX_WIDTH_SLICE, PLT_KEYS.AX_HEIGHT_SLICE)
    ) 
    ax = _raw_cell_types(None, x, y, ax, celltypes, bg.label_to_col, s = PLT_KEYS.SIZE_PT_CELL_ONSLICE)

    fig.tight_layout()

    # SAVE
    output_name = PLT_KEYS.FOLDER_NAME + '/' + PLT_KEYS.FILE_PREFIX_RAWSLICE + 'labels' + PLT_KEYS.FILE_FORMAT
    fig.savefig(output_name, bbox_inches = 'tight')

def Plot_Spots(
    bg: Optional[BrGraph] = None,
    df_spots_seg: Optional[pd.DataFrame] = None,
    df_spots_unseg: Optional[pd.DataFrame] = None,
):
    '''
    Visualize both segmented and unsengmented spots

    Parameters
    ----------
    bg: BrGraph
        Bering Graph object. 
        - If bg is not None, then df_spots_seg and df_spots_unseg are ignored.
        - If bg is None, then df_spots_seg and df_spots_unseg must be provided.
    df_spots_seg: pd.DataFrame
        DataFrame of segmented spots
    df_spots_unseg: pd.DataFrame
        DataFrame of unsegmented spots
    '''

    if bg is not None:
        df_spots_seg = bg.spots_seg.copy()
        df_spots_unseg = bg.spots_unseg.copy()

    x, y = df_spots_seg['x'].values, df_spots_seg['y'].values
    cell_types = df_spots_seg['labels'].values

    fig, ax = plt.subplots(figsize = (5, 5))
    for idx, cell_type in enumerate(np.unique(cell_types)):
        
        xc = x[np.where(cell_types == cell_type)[0]]
        yc = y[np.where(cell_types == cell_type)[0]]

        ax.scatter(xc, yc, s = 0.03, label = cell_type, color = np.random.rand(3,))

    xb, yb = df_spots_unseg['x'].values, df_spots_unseg['y'].values
    ax.scatter(xb, yb, color = '#DCDCDC', alpha = 0.2, s = 0.015, label = 'background')

    h, l = ax.get_legend_handles_labels()
    plt.legend(h, l, loc = 'upper right', fontsize = 8, markerscale = 15)

def Plot_Classification(
    bg: BrGraph, 
    cell_name: str, 
    n_neighbors: int = 10, 
    min_prob: float = 0.3, 
    zoomout_scale: float = 8.0,
):
    '''
    Plot node classfication results on the original data and predicted data.
    
    Parameters
    ----------
    bg: BrGraph
        Bering Graph object
    cell_name: str
        Name of the cell to plot
    n_neighbors: int
        Number of neighbors to build the knn graph
    min_prob: float
        Minimum probability threshold to classify a valid cell type (otherwise background)
    zoomout_scale: float
        Zoom out scale (relative to the cell diameter) to show the region
    
    Returns
    -------
    - df_window_raw: ``pd.DataFrame``
        DataFrame of the raw spots in the window
    - df_window_pred: ``pd.DataFrame``
        DataFrame of the predicted spots in the window
    - predictions: ``np.ndarray``
        Array of predicted labels
    '''
    # BUILD GRAPHS
    if len(bg.label_to_col) == 0:
        bg.label_to_col = dict(zip(bg.labels, CMAP[:bg.n_labels]))
        bg.label_to_col['background'] = '#C0C0C0'

    Spots = bg.spots_all.copy()
    cell_metadata = bg.segmented.copy()

    df_window_raw, graph = _get_graph(
        bg, cell_name, Spots, cell_metadata, n_neighbors = n_neighbors, zoomout_scale_x = zoomout_scale, zoomout_scale_y = zoomout_scale,
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

    seg_types = Spots.loc[tps, 'groups'].values
    raw_labels = Spots.loc[tps, 'labels'].values
    pred_labels = np.array([bg.label_indices_dict[i] for i in predictions])
    label_to_col = bg.label_to_col

    accuracy = np.sum(raw_labels == pred_labels) / len(raw_labels)

    # PLOTS
    fig, axes = plt.subplots(figsize = (PLT_KEYS.AX_WIDTH * 2, PLT_KEYS.AX_HEIGHT *2), ncols = 2, nrows = 2, sharex = True, sharey = True) 
    if bg.image_raw is not None:
        dapi = bg.image_raw[0,:,:]
    else:
        dapi = None
    axes[0, 0] = _raw_segmentation(dapi, x, y, axes[0,0], seg_types)
    axes[0, 1] = _raw_cell_types(dapi, x, y, axes[0, 1], raw_labels, label_to_col)
    axes[1, 0] = _predicted_cell_types(dapi, x, y, axes[1, 0], pred_labels, label_to_col)
    axes[1, 1], prob_plot = _predicted_probability(dapi, x, y, axes[1, 1], max_probs)
    axes[1, 0].set_title(f'Predicted Annotations (Accu={accuracy:.2f})')
    plt.colorbar(prob_plot, ax = axes[1,1], shrink = 0.8)
    fig.tight_layout()

    # SAVE
    output_name = PLT_KEYS.FOLDER_NAME + '/' + PLT_KEYS.FILE_PREFIX_CLASSIFICATION + str(cell_name) + PLT_KEYS.FILE_FORMAT
    fig.savefig(output_name, bbox_inches = 'tight')

    # FOREGROUND WINDOW
    df_window_pred = df_window_raw.iloc[np.where(predictions != bg.n_labels - 1)[0],:].copy()
    return df_window_raw, df_window_pred, predictions

def Plot_Segmentation(
    bg: BrGraph, 
    cell_name: str,
    df_window_raw: pd.DataFrame = None, 
    df_window_pred: pd.DataFrame = None,
    predictions: np.ndarray = None,
    n_neighbors: int = 10, 
    zoomout_scale: float = 4.0,
    use_image: bool = True,
    pos_thresh: float = 0.6,
    resolution: float = 0.05,
    num_edges_perSpot: int = 300,
    min_prob_nodeclf: float = 0.3,
    n_iters: int = 10,
    convex_hull: bool = False,
):
    '''
    Plot the segmentation results with cell IDs on the original data and predicted data.

    Parameters
    ----------
    bg: BrGraph
        Bering Graph object
    cell_name: str
        Name of the cell to plot
    df_window_raw: pd.DataFrame
        DataFrame of the raw spots in the window
    df_window_pred: pd.DataFrame
        DataFrame of the predicted spots in the window
    predictions: np.ndarray
        Array of predicted labels. This is used to identify the foreground spots.
    n_neighbors: int
        Number of neighbors to build the knn graph
    zoomout_scale: float
        Zoom out scale (relative to the cell diameter) to show the region
    use_image: bool
        Whether to use the image to build the graph
    pos_thresh: float
        Threshold to determine whether the predicted edge is positive in the segmentation step
    resolution: float
        Resolution of Leiden clustering algorithm in the segmentation step
    num_edges_perSpot: int
        Number of nearest edges used to investigate edge labels (positive or negative) for each spot
    min_prob_nodeclf: float
        Minimum probability threshold to classify a valid cell type (otherwise background)
    n_iters: int
        Number of iterations. Each iteration runs on a subset of edges. This is used to avoid memory overflow.
    convex_hull: bool
        Whether to use convex hull to draw the cells
    '''
    Spots = bg.spots_all.copy()
    cell_metadata = bg.segmented.copy()

    if df_window_raw is None:
        # clean background nodes
        df_window_raw, graph = _get_graph(
            bg, cell_name, Spots, cell_metadata, n_neighbors = n_neighbors, zoomout_scale_x = zoomout_scale, zoomout_scale_y = zoomout_scale, 
        )
        graph = graph.cpu()
        predictions, max_probs = _prediction_nodes(graph, bg.trainer_node, bg.n_labels, prob_threshold = min_prob_nodeclf)
        predictions = predictions.numpy()
        max_probs = max_probs.numpy()

        df_window_pred = df_window_raw.iloc[np.where(predictions != bg.n_labels - 1)[0], :].copy()
    
    cells_raw = df_window_raw.segmented.values
    cells_pred = cells_raw.copy()

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
    cells_pred[np.where(predictions != bg.n_labels - 1)[0]] = cls_predlink
    cells_pred[np.where(predictions == bg.n_labels - 1)[0]] = -1
    
    # get labels
    raw_labels = df_window_raw.labels.values
    pred_labels = np.array([bg.label_indices_dict[i] for i in predictions])
    

    # ARI_score_predlink = adjusted_rand_score(cells_raw, cells_pred)
    x_raw, y_raw = df_window_raw.x.values, df_window_raw.y.values
    x_predfore, y_predfore = df_window_pred.x.values, df_window_pred.y.values

    if bg.channels is not None:
        fig, axes = plt.subplots(figsize = (PLT_KEYS.AX_WIDTH * (bg.n_channels + 1) * 0.8, PLT_KEYS.AX_HEIGHT * 0.8), ncols = bg.n_channels + 1, nrows = 1, sharex = True, sharey = True)
        for idx in range(bg.n_channels):
            if convex_hull:
                axes[idx] = _draw_cells_withStaining_convexhull(bg.image_raw[idx], x_raw, y_raw, axes[idx], cells_raw, raw_labels, bg.label_to_col, 'Raw (w. ' + bg.channels[idx] + ')' )
            else:
                axes[idx] = _draw_cells_withStaining(bg.image_raw[idx], x_raw, y_raw, axes[idx], cells_raw, 'Raw (w. ' + bg.channels[idx] + ')')

        if convex_hull:
            axes[idx] = _draw_cells_withStaining_convexhull(bg.image_raw[0], x_predfore, y_predfore, axes[-1], cls_predlink, pred_labels, bg.label_to_col, 'Pred (w. DAPI)')
        else:
            axes[-1] = _draw_cells_withStaining(bg.image_raw[0], x_predfore, y_predfore, axes[-1], cls_predlink, 'Pred (w. DAPI)')

    else:
        fig, axes = plt.subplots(figsize = (PLT_KEYS.AX_WIDTH * 2 * 0.8, PLT_KEYS.AX_HEIGHT * 0.8), ncols = 2, nrows = 1, sharex = True, sharey = True)
        if convex_hull:
            axes[0] = _draw_cells_withStaining_convexhull(None, x_raw, y_raw, axes[0], cells_raw, raw_labels, bg.label_to_col, 'Raw')
        else:
            axes[0] = _draw_cells_withStaining(None, x_raw, y_raw, axes[0], cells_raw, 'Raw')

        # axes[1] = _draw_cells_withStaining(None, x_predfore, y_predfore, axes[1], cls_predlink, 'Pred (ARI=' + str(np.round(ARI_score_predlink,2)) + ')')
        if convex_hull:
            axes[1] = _draw_cells_withStaining_convexhull(None, x_predfore, y_predfore, axes[1], cls_predlink, pred_labels, bg.label_to_col, 'Predicted')
        else:
            axes[1] = _draw_cells_withStaining(None, x_predfore, y_predfore, axes[1], cls_predlink, 'Predicted')

    fig.tight_layout()

    output_name = PLT_KEYS.FOLDER_NAME + '/' + PLT_KEYS.FILE_PREFIX_SEGMENTATION + str(cell_name) + PLT_KEYS.FILE_FORMAT
    fig.savefig(output_name, bbox_inches = 'tight')