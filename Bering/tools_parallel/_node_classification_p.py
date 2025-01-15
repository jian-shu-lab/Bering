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

def node_classification_p(
    bg: BrGraph,
    df_spots: pd.DataFrame,
    n_neighbors: int = 10,
    beta: float = 1.0,
    prob_threshold: float = 0.3, 
):
    '''
    Node classification for provided spots table (spots for each FOV is recommended with ~ 10,000 spots)

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

    Returns
    -------
    preds_labels: np.array
        predicted labels for all spots
    graph_all: torch_geometric.data.Data
        graph (``torch_geometric.data.Data`` object) for the whole slice
    '''

    # build graph
    logger.info(f'Runnning node classification ...')
    # try:
        # graph_all = BuildGraph_fromRaw(bg, df_spots, bg.features.copy(), n_neighbors, beta).cpu()
        # bg.graph_all = graph_all
    # except:
    #     graph_all = None
    #     bg.pos_all = _get_pos(bg.spots_all)
    #     logger.info(f'    There are too many spots in the slice. Skip the generation of the whole graph.')

    bg.spots_all = df_spots.copy()
    graph_all = BuildGraph_fromRaw(bg, df_spots, bg.features.copy(), n_neighbors, beta, device = 'cpu').cpu()
    bg.graph_all = graph_all

    # get latent
    logger.info(f'    Get the latent space for all {df_spots.shape[0]} nodes')
    bg.trainer_node.model.to('cpu')
    bg.z_all = bg.trainer_node.model.get_latent(graph_all)
    preds_logits = bg.trainer_node.predict(graph_all, device = 'cpu').cpu()

    # prediction results
    max_probs, preds_logits = torch.max(preds_logits, dim = 1)
    back_indices = torch.where(max_probs <= prob_threshold)[0].unsqueeze(1)
    preds_logits[back_indices] = bg.n_labels_raw - 1

    preds_labels = np.array([bg.label_indices_dict[i.item()] for i in preds_logits])
    bg.spots_all['predicted_labels'] = preds_labels
    bg.spots_all['predicted_probability'] = max_probs.numpy()

    bg.foreground_indices = np.where(bg.spots_all['predicted_labels'].values != 'background')[0]

    # output results
    return preds_labels, graph_all