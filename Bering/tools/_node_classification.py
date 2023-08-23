import logging
import torch
import numpy as np
import pandas as pd

from ..objects import Bering_Graph as BrGraph
from ..graphs import BuildGraph_fromRaw

logger = logging.getLogger(__name__)

def _get_pos(
    df_spots: pd.DataFrame,
):
    x = df_spots.x.values
    y = df_spots.y.values
    z = df_spots.z.values
    tps_names = df_spots.index.values

    # pos = np.array([tps_names, x, y]).T
    pos = np.array([tps_names, x, y, z]).T
    pos = torch.from_numpy(pos).double()
    return pos

def _get_node_embedding_prediction_byTiling(bg, df_spots, num_chunks, n_neighbors, beta, dummy = 1e-3):
    '''
    Split spots tables into tiles by coordinates if the number of spots is too large (use 2d chunks here)
    '''
    x, y = df_spots.x.values, df_spots.y.values
    num_chunks_axis = np.round(np.sqrt(num_chunks)).astype(int)
    num_chunks = num_chunks_axis ** 2
    logger.info(f'Number of chunks for node classification (adjusted): {num_chunks}')

    # tile_size_x = (np.max(x) - np.min(x)) / num_chunks_axis
    # tile_size_y = (np.max(y) - np.min(y)) / num_chunks_axis
    x_percentiles = np.percentile(x, np.linspace(0, 100, num_chunks_axis + 1))
    y_percentiles = np.percentile(y, np.linspace(0, 100, num_chunks_axis + 1))

    if not hasattr(bg.trainer_node.model, 'num_mlp_layers_remain'):
        z_all = torch.zeros((df_spots.shape[0], bg.trainer_node.model.mlp_layer_dims[1]), dtype = torch.double) # GCN
    else:
        z_all = torch.zeros((df_spots.shape[0], bg.trainer_node.model.mlp_layer_dims[-(bg.trainer_node.model.num_mlp_layers_remain+1)]), dtype = torch.double) # MLP
    
    logger.info(f'size of z_all: {z_all.shape}')
    preds_logits = torch.zeros((df_spots.shape[0], bg.n_labels), dtype = torch.double)

    for i in range(num_chunks_axis):
        for j in range(num_chunks_axis):
            # min_x, max_x = np.min(x) + i * tile_size_x, np.min(x) + (i + 1) * tile_size_x
            # min_y, max_y = np.min(y) + j * tile_size_y, np.min(y) + (j + 1) * tile_size_y
            min_x, max_x = x_percentiles[i], x_percentiles[i + 1]
            min_y, max_y = y_percentiles[j], y_percentiles[j + 1]

            if i == 0:
                min_x -= dummy
            if i == num_chunks_axis - 1:
                max_x += dummy
            if j == 0:
                min_y -= dummy
            if j == num_chunks_axis - 1:
                max_y += dummy

            tile_indices = np.where(
                (x >= min_x) & (x < max_x) & (y >= min_y) & (y < max_y)
            )[0]
            # logger.info(f'tile_indices (top 10 and bottom 10): {tile_indices[:10]}, {tile_indices[-10:]}')

            if len(tile_indices) > n_neighbors + 1:
                df_spots_section = df_spots.iloc[tile_indices, :]
                logger.info(f'Number of spots in tile (i,j) {(i,j)} is {df_spots_section.shape[0]}')
                graph_section = BuildGraph_fromRaw(bg, df_spots_section, bg.features.copy(), n_neighbors, beta).cpu()
                z_section = bg.trainer_node.model.get_latent(graph_section)
                logger.info(f'size of z_section (1): {z_section.shape}')
                preds_logits_section = bg.trainer_node.predict(graph_section, device = 'cpu').cpu()
            else:
                z_section = torch.zeros((len(tile_indices), z_all.shape[1]), dtype = torch.double)
                logger.info(f'size of z_section (2): {z_section.shape}')
                preds_logits_section = torch.zeros((len(tile_indices), bg.n_labels), dtype = torch.double)

            # if i == 0 and j == 0:
            #     z_all = z_section
            #     preds_logits = preds_logits_section
            # else:
            #     z_all = torch.cat((z_all, z_section), dim=0)
            #     preds_logits = torch.cat((preds_logits, preds_logits_section), dim=0)

            z_all[tile_indices, :] = z_section
            preds_logits[tile_indices, :] = preds_logits_section
    
    return z_all, preds_logits

def node_classification(
    bg: BrGraph,
    df_spots: pd.DataFrame,
    n_neighbors: int = 10,
    beta: float = 1.0,
    prob_threshold: float = 0.3, 
    max_num_spots: int = 1500000, #1.5 million
    num_chunks: int = 25,
):
    '''
    Node classification for all spots in the slice

    Parameters
    ----------
    bg: BrGraph
        Bering Graph object
    df_spots: pd.DataFrame
        spots table. It can be ``bg.spots_all`` in case of whole slice prediction
    n_neighbors: int
        number of neighbors for graph construction
    prob_threshold: float
        minimal threshold of predicted probability for spots to be considered as foreground
    max_num_spots: int
        maximum number of spots for node classification in each chunk. 
        If the number of spots is larger than this number, the spots table will be split into chunks by coordinates.
    num_chunks: int
        number of chunks for node classification. This is done by splitting the spots table into chunks by coordinates.
        This is used when the number of spots is too large. Refer to `_get_node_embedding_prediction_byTiling` for details.

    Returns
    -------
    preds_labels: np.array
        predicted labels for all spots
    graph_all: torch_geometric.data.Data
        graph (``torch_geometric.data.Data`` object) for the whole slice
    '''

    # build graph
    logger.info(f'Building Graph for the whole slice')
    try:
        graph_all = BuildGraph_fromRaw(bg, df_spots, bg.features.copy(), n_neighbors, beta).cpu()
        bg.graph_all = graph_all
    except:
        graph_all = None
        bg.pos_all = _get_pos(bg.spots_all)
        logger.info(f'There are too many spots in the slice. Skip the generation of the whole graph.')

    # get latent
    logger.info(f'Get the latent space for all {df_spots.shape[0]} nodes')
    bg.trainer_node.model.to('cpu')
    if bg.spots_all.shape[0] <= max_num_spots:
        bg.z_all = bg.trainer_node.model.get_latent(graph_all)
        preds_logits = bg.trainer_node.predict(graph_all, device = 'cpu').cpu()
    else:
        logger.info(f'Number of chunks for node classification: {num_chunks}')
        bg.z_all, preds_logits = _get_node_embedding_prediction_byTiling(bg, df_spots, num_chunks, n_neighbors, beta)

    # prediction results
    max_probs, preds_logits = torch.max(preds_logits, dim = 1)
    back_indices = torch.where(max_probs <= prob_threshold)[0].unsqueeze(1)
    preds_logits[back_indices] = bg.n_labels - 1

    preds_labels = np.array([bg.label_indices_dict[i.item()] for i in preds_logits])
    bg.spots_all['predicted_node_labels'] = preds_labels
    bg.spots_all['predicted_probability'] = max_probs.numpy()

    bg.foreground_indices = np.where(bg.spots_all['predicted_node_labels'].values != 'background')[0]

    # output results
    return preds_labels, graph_all