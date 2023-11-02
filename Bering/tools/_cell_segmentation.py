import logging
import numpy as np
import pandas as pd

from ..objects import Bering_Graph as BrGraph
from ..segment import find_clusters_predictedLinks

logger = logging.getLogger(__name__)

def cell_segmentation(
    bg: BrGraph,
    use_image: bool = True,
    positive_edge_thresh: float = 0.6,
    leiden_resolution: float = 1e-2,
    num_edges_perSpot: int = 300,
    graph_n_neighbors: int = 10,
    num_iters: int = 100,
):
    '''
    Run cell segmentation for all spots.

    Parameters
    ----------
    bg: BrGraph
        Bering graph object
    use_image: bool
        Whether to use image features for cell segmentation
    positive_edge_thresh: float
        Minimal threshold of prediction probability for edges to be considered as positive
    leiden_resolution: float
        Resolution for Leiden clustering in the segmentation step
    num_edges_perSpot: int
        Number of nearest edges for each spot to be considered in the graph construction
    graph_n_neighbors: int
        Number of neighbors for graph construction if ``bg.graph_all`` == ``None`` and ``bg.z_all`` == ``None``
    num_iters: int
        Number of iterations for edge label prediction. This is only used when number of edges are to large.
        It can avoid memory overflow.
    '''
    logger.info(f'Running cell segmentation ...')
    # predicted_cells = np.array(['background'] * bg.spots_all.shape[0])
    predicted_cells = np.array([0] * bg.spots_all.shape[0])
    foreground_indices = np.where(bg.spots_all['predicted_labels'].values != 'background')[0]
    bg.foreground_indices = foreground_indices
    df_spots = bg.spots_all.iloc[foreground_indices, :].copy()
    if use_image and bg.image_raw is not None:
        split_by_tiling = True
    else:
        split_by_tiling = False
    
    if leiden_resolution is None:
        leiden_resolutions = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    else:
        if not isinstance(leiden_resolution, list):
            leiden_resolutions = [leiden_resolution]

    clusters = find_clusters_predictedLinks(
        bg = bg,
        df_spots = df_spots, 
        use_image = use_image, 
        pos_thresh = positive_edge_thresh,
        resolutions = leiden_resolutions,
        num_edges_perSpot = num_edges_perSpot,
        n_neighbors = graph_n_neighbors,
        num_iters = num_iters,
        split_edges_byTiling = split_by_tiling,
    )
    # for k, v in clusters.items():
    #     predicted_cells[foreground_indices] = v
    # bg.spots_all['predicted_cells'] = predicted_cells
    predicted_cells[foreground_indices] = clusters
    bg.spots_all['predicted_cells'] = predicted_cells
    
    # return clusters