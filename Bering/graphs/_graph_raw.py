import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn.functional as F 
from torch_geometric.data import Data 

from ..objects import Bering_Graph as BrGraph

def _get_edge_index(
    df_spots: pd.DataFrame,
    df_features: pd.DataFrame,
    n_neighbors: int,
):
    '''
    Get edges using KNN graph

    Parameters
    ----------
    bg
        Bering object
    df_spots
        The table of spots from neighboring cells
    n_neighbors
        Number of neighbors in KNN
    '''
    x = df_spots.x.values
    y = df_spots.y.values
    z = df_spots.z.values
    tps_features = df_spots.features.values
    
    features = df_features.index.values
    features_indices = np.arange(len(features)) # 1-1 match to features
    feature_dict = dict(zip(features, features_indices))
    tps_features_indices = np.array([feature_dict[i] for i in tps_features])
    
    ### edges
    coords = np.array([x, y, z]).T 
    A = kneighbors_graph(
        coords, n_neighbors = n_neighbors, mode = 'distance'
    ) # Adjacency
    A = coo_matrix(A)
    E = np.array((A.row, A.col), dtype = np.int64)
    E = torch.from_numpy(E)

    return A, E, tps_features_indices

def _get_edge_attr(
    A, beta
):
    E_attr = np.exp(-beta * A.data / A.data.mean())
    E_attr = torch.from_numpy(E_attr)
    return E_attr

def _get_node_features(
    bg,
    df_spots: pd.DataFrame,
    n_features: int,
    A = None, # sparse matrix
    tps_features_indices = None,
    n_neighbors = 10,
    use_coordinates = False,
    use_transcriptomics = True, 
    use_morphological = True,
):
    '''
    Generate node features. Currently they include:
    (1) Location: 2d coordinates
    (2) Transcriptomic: neighboring transcriptomic components
    (3) Morphological: density of nodes in the region

    Parameters
    ----------
    df_spots
        The table of spots from neighboring cells
    A
        Sparse adjancy matrix of the nodes (coo_matrix)
    tps_features_indices
        Indices for transcript features
    '''
    n_tps = df_spots.shape[0]
    x, y, z = df_spots.x.values, df_spots.y.values, df_spots.z.values
    X_coords = None; X_trans = None; X_morpho = None

    if use_coordinates:
        X_coords = np.array([x, y, z]).T

    if use_transcriptomics:
        X_trans = coo_matrix((np.ones(A.data.shape[0]), (A.row, tps_features_indices[A.col])), shape = (n_tps, n_features)).toarray().astype(np.float32)

    if use_morphological:
        X_morpho = A.sum(axis = 1).A1.reshape((n_tps, 1))
        X_morpho = X_morpho / n_neighbors

    # concat
    X_vector = []
    for i in [X_coords, X_trans, X_morpho]:
        if i is not None:
            X_vector.append(i)

    X = np.concatenate(X_vector, axis = 1)
    X = torch.from_numpy(X).double()

    return X

def _get_pos(
    df_spots: pd.DataFrame,
):
    '''
    Add additional information (2d coordinates and cell ids) 
    of nodes into data.pos

    Parameters
    ----------
    df_spots
        Table of spots from neighboring cells
    df_cells
        Metadata table of cells
    '''
    x = df_spots.x.values
    y = df_spots.y.values
    z = df_spots.z.values
    tps_names = df_spots.index.values

    pos = np.array([tps_names, x, y, z]).T
    pos = torch.from_numpy(pos).double()
    return pos

def BuildGraph_fromRaw(
    bg: BrGraph, 
    df_spots: pd.DataFrame,
    df_features: pd.DataFrame,
    n_neighbors: int = 10,
    beta: float = 1.0,
    device: str = 'cuda',
):
    '''
    Build Graphs from unsegmented spots

    Parameters
    ----------
    bg
        Bering object
    df_spots
        input dataframe of spots with columns of x, y and features
    df_features
        the metadata table of all features
    n_neighbors
        number of neighbors in KNN
    device
        device to run the model. Options: 'cuda' or 'cpu'
    
    Returns
    -------
    Graph of ``torch_geometric.data.Data``
    '''
    
    # get edges
    A, E, tps_features_indices = _get_edge_index(df_spots, df_features, n_neighbors)
    # get node features
    X = _get_node_features(bg, df_spots, df_features.shape[0], A, tps_features_indices, n_neighbors)
    # get edge attributes
    E_attr = _get_edge_attr(A, beta)
    # get node additional information
    POS = _get_pos(df_spots)

    # create graph
    G = Data(x = X, edge_index = E, edge_attr = E_attr, pos = POS)
    G = G.to(device)

    return G