import logging
import collections
import torch
import torch.nn.functional as F 

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from ._gnn import GCN

logger = logging.getLogger(__name__)

def _sample_edges(
    data: DataLoader, 
    num_pos_edges: int, 
    num_neg_edges: int, 
    num_edges_perG: int = 10000
):
    '''
    Sample random positive & negative edges as the input of decoder.

    Parameters
    ----------
    data
        Input data loader; 
        Node coordinates and cell ids are stored in data.pos
    num_pos_edges
        Number of positive edges
    num_neg_edges
        Number of negative edges
    num_edges_perG
        Number of edges per graph
    '''

    # init
    N = data.pos.shape[0]
    cells = data.pos[:,4]

    # get all edges
    if hasattr(data, "ptr"):
        batch_size = len(data.ptr) - 1
        num_nodes = data.ptr[1:] - data.ptr[:-1]
        num_cum_nodes = data.ptr
    else:
        batch_size = 1
        num_nodes = torch.tensor([N], dtype=torch.long, device="cuda")
        num_cum_nodes = torch.tensor([0, N], dtype=torch.long, device="cuda")

    # get edges per graph
    edges = torch.rand((2, batch_size, num_edges_perG), dtype=torch.float, device="cuda")     # for each graph, we sample num_edges_perG edges
    edges_graph_indices = torch.LongTensor(
        list(range(batch_size))).repeat_interleave(edges.shape[-1]).reshape(1,batch_size,edges.shape[-1]
    ).cuda() # add graph indices to each edge

    edges = (edges * num_nodes[None, :, None]).long()
    edges = edges + num_cum_nodes[None, :-1, None]
    edges = torch.cat([edges, edges_graph_indices], dim = 0) # 3 x batch_size x num_edges
    edges = edges.flatten(-2, -1)
    edges = edges[:, edges[0] != edges[1]]

    # get all labels
    srcs = edges[0]; dsts = edges[1]

    pos_edge_index = edges[:, torch.where((cells[srcs] == cells[dsts]) & (cells[srcs] != -1) & (cells[dsts] != -1))[0]]
    neg_edge_index = edges[:, torch.where((cells[srcs] != cells[dsts]) & (cells[srcs] != -1) & (cells[dsts] != -1))[0]]

    # ensure equal number of pos / neg edges
    num_min_edges = min(pos_edge_index.shape[1], neg_edge_index.shape[1])
    num_pos_edges = min(num_min_edges, num_pos_edges)
    num_neg_edges = min(num_min_edges, num_neg_edges)
    logger.info(
        f'Numer of total positive edges: {pos_edge_index.shape[1]}; negative edges: {neg_edge_index.shape[1]}; final edges for pos/neg types: {num_pos_edges}, {num_neg_edges}'
    )

    # format pos/neg edges
    pos_edge_index = pos_edge_index[:, torch.randint(0, pos_edge_index.shape[1], (num_pos_edges,), dtype = torch.long)]
    pos_edge_graph_indices = torch.cat([pos_edge_index[2], pos_edge_index[2]])
    pos_edge_index = torch.cat([pos_edge_index[:2, :], pos_edge_index[[1,0], :]], dim = -1)

    neg_edge_index = neg_edge_index[:, torch.randint(0, neg_edge_index.shape[1], (num_neg_edges,), dtype = torch.long)]
    neg_edge_graph_indices = torch.cat([neg_edge_index[2], neg_edge_index[2]])
    neg_edge_index = torch.cat([neg_edge_index[:2, :], neg_edge_index[[1,0], :]], dim = -1)

    edge_index = torch.cat((pos_edge_index, neg_edge_index), axis = -1)
    edge_labels = torch.cat((
        torch.ones((pos_edge_index.shape[1], ), dtype = torch.float64, device = 'cuda'),
        torch.zeros((neg_edge_index.shape[1], ), dtype = torch.float64, device = 'cuda'),
    ))
    edge_graph_index = torch.cat([pos_edge_graph_indices, neg_edge_graph_indices])
    
    return edge_index, edge_labels, edge_graph_index

def _binning_coordinates(
    src_coords_x: torch.Tensor,
    dst_coords_x: torch.Tensor,
    image_binsize: int = 5,
    min_image_size: int = 5,
    max_image_size: int = 40, 
):
    '''
    Binning the distance between source and destination nodes in 1d (either x or y)
    Input: source and destination node coordinates in height (or width) with shape (n,)
    Output: minimal and maximal coordinates in height (or width) for source and destination. New coordinates after binning

    Parameters
    ----------
    src_coords_x
        1d coordinates of source nodes
    dst_coords_x
        1d coordinates of destination nodes
    image_binsize
        bin size of the image in height and width
    min_image_size
        minimal image size in height and width
    max_image_size
        maximal image size in height and width
    '''
    # input src coordinates (n x 2) and dst coordinates (n x 2); all are integars

    # get min max values
    minx = torch.min(src_coords_x, dst_coords_x)
    maxx = torch.max(src_coords_x, dst_coords_x)

    dist_x = maxx - minx

    # small distances
    smalldist_indices = torch.where(dist_x < image_binsize)[0]
    smalldist = dist_x[smalldist_indices]
    gap_to_minimal = min_image_size - smalldist
    gap_half1 = gap_to_minimal // 2
    gap_half2 = gap_to_minimal - gap_half1
    minx[smalldist_indices] = minx[smalldist_indices] - gap_half1
    maxx[smalldist_indices] = maxx[smalldist_indices] + gap_half2

    dist_x = maxx - minx

    # binning
    dist_x_appro = torch.round(torch.clip(dist_x, min_image_size, max_image_size) / image_binsize).long() * image_binsize
    gap_x = dist_x_appro - dist_x
    gap_half1 = gap_x // 2
    gap_half2 = gap_x - gap_half1
    
    minx = minx - gap_half1
    maxx = maxx + gap_half2

    return minx, maxx, dist_x_appro