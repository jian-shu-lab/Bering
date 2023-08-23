import igraph
import logging
import numpy as np
import pandas as pd

import leidenalg as la
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph

import torch
import torch.nn.functional as F 
from torch_geometric.data import Data 

from ..objects import Bering_Graph as BrGraph
from ..training import TrainerEdge
from ..graphs import BuildGraph_fromRaw
from ..models import _get_image_graph, _get_binned_coordinates

logger = logging.getLogger(__name__)
from ._settings import SEGMENT_KEYS as SEG_KEYS

@torch.no_grad()
def _get_edge_embedding(
    trainer: TrainerEdge,
    pos: torch.Tensor,
    image: torch.Tensor,
    z_node: torch.Tensor,
    edge_indices: torch.Tensor,
    max_subplots_perRound: int = 5000,
):
    '''
    Get edge embeddings for selected edges
    '''
    # get node and RBF features
    trainer.model.decoder.eval()
    if trainer.model.distance_type == 'rbf':
        trainer.model.rbf_kernel.eval()
    if trainer.model.encoder_image is not None:
        trainer.model.encoder_image.eval()

    if trainer.model.distance_type == 'positional':
        # z_node = torch.cat((z_node, pos[:,[1,2]]), axis = -1) # 2d
        z_node = torch.cat((z_node, pos[:,[1,2,3]]), axis = -1) # 3d
    src = edge_indices[:,0]
    dst = edge_indices[:,1]

    edge_attr = torch.cat((z_node[src], z_node[dst]), axis = -1)
    # src_coords = pos[src, :][:,[1,2]] # 2d
    # dst_coords = pos[dst, :][:,[1,2]]
    src_coords = pos[src, :][:,[1,2,3]] # 3d
    dst_coords = pos[dst, :][:,[1,2,3]]

    if trainer.model.distance_type == 'rbf':
        trainer.model.rbf_kernel.to('cpu')
        edge_attr_rbf = trainer.model.rbf_kernel(x = src_coords, y = dst_coords)           
        edge_attr = torch.cat((edge_attr, edge_attr_rbf), axis = -1)
        del edge_attr_rbf

    pos = torch.cat((pos[src,:], pos[dst,:]), axis = 0)

    # get image features
    if (image is not None) and trainer.model.image_model:
        import time
        # get conv2d embeddings
        t0 = time.time()
        image_graph, src_coords, dst_coords = _get_image_graph(pos, image, src_coords, dst_coords)
        image_graph = trainer.model.encoder_image.get_conv2d_embedding(image_graph)
        t1 = time.time()
        logger.info(f'---Get image graph time: {(t1-t0):.5f} s. Number of edges: {src_coords.shape[0]}')
        
        # binning coordinates
        minx, maxx, miny, maxy, avail_bins, dist_bins_2d = _get_binned_coordinates(
            src_coords, dst_coords, trainer.model.image_binsize, trainer.model.min_image_size, trainer.model.max_image_size
        )
        t2 = time.time()
        logger.info(f'---Get all binned coordinates time: {(t2-t1):.5f} s')

        # run the model for eachedge
        edge_attr_image = torch.empty((src_coords.shape[0], trainer.model.n_image_features)).double().cuda()
        for avail_bin in avail_bins:
            bin_indices = torch.where((dist_bins_2d == avail_bin).all(dim=1))[0]
            logger.info(f'---Number of edges in bin {avail_bin}: {len(bin_indices)}')
            subimages = []
            _max_subplots_perRound = int(min(5*1024*1024*1024/(2048*avail_bin[0]*avail_bin[1]), max_subplots_perRound)) # no larger than 5BG
            if len(bin_indices) < _max_subplots_perRound:
                for i,j in enumerate(bin_indices):
                    subimage = image_graph[:,:,miny[j]:maxy[j], minx[j]:maxx[j]]
                    subimages.append(subimage)
                            
                subimages = torch.cat(subimages, axis = 0)
                edge_attr_image_bin = trainer.model.encoder_image(subimages)
                edge_attr_image[bin_indices, :] = edge_attr_image_bin
            else:
                for i in range(0, len(bin_indices), _max_subplots_perRound):
                    subimages = []
                    for j in bin_indices[i:i+_max_subplots_perRound]:
                        subimage = image_graph[:,:,miny[j]:maxy[j], minx[j]:maxx[j]]
                        subimages.append(subimage)
                    subimages = torch.cat(subimages, axis = 0)
                    edge_attr_image_bin = trainer.model.encoder_image(subimages)
                    edge_attr_image[bin_indices[i:i+_max_subplots_perRound], :] = edge_attr_image_bin
        t3 = time.time()
        logger.info(f'---Get all image embeddings time: {(t3-t2):.5f} s')

        edge_attr_image = edge_attr_image.cpu()
        edge_attr = torch.cat([edge_attr, edge_attr_image], dim = -1)

    return edge_attr, edge_indices

@torch.no_grad()
def _get_edge_labels(
    bg,
    edge_attr,
):
    '''
    Get edge labels with edge attributes as the input
    '''
    bg.trainer_edge.model.decoder.eval()
    if bg.trainer_edge.model.distance_type == 'rbf':
        bg.trainer_edge.model.rbf_kernel.eval()
    if bg.trainer_edge.model.encoder_image is not None:
        bg.trainer_edge.model.encoder_image.eval()
    bg.trainer_edge.model.decoder.to('cpu')
    pred_logits = bg.trainer_edge.model.decoder(edge_attr)
    pred_logits = F.sigmoid(pred_logits).squeeze().numpy()
    return pred_logits

def _create_adjacency_matrix(
    edge_indices,
    pred_prob,
    pos_thresh,
    neg_thresh,
    N_nodes,
):
    '''
    Define adjacency matrix graph
    '''
    pos_edges = np.where(pred_prob > pos_thresh)[0] # define positive edges
    neg_edges = np.where(pred_prob <= neg_thresh)[0] # define negative edges
    
    n_pos = len(pos_edges)
    n_neg = len(neg_edges)
    total = pred_prob.shape[0]
    logger.info(f'Total Number of nodes: {N_nodes}; Total number of edges: {total}; Prediction Probability Range: {(np.min(pred_prob)):.3f} ~ {(np.max(pred_prob)):.3f}')
    logger.info(f'  Positive edge (prob >= {pos_thresh:.3f}) ratio: {(n_pos / total):.3f}')
    logger.info(f'  Negative edge (prob < {neg_thresh:.3f}) ratio: {(n_neg / total):.3f}')

    pos_edge_indices = edge_indices[pos_edges]
    neg_edge_indices = edge_indices[neg_edges]
    pred_prob = pred_prob[pos_edges]

    row = np.concatenate((pos_edge_indices[:,0], pos_edge_indices[:,1]))
    col = np.concatenate((pos_edge_indices[:,1], pos_edge_indices[:,0]))
    data = np.concatenate((pred_prob, pred_prob))

    g = igraph.Graph()
    g.add_vertices(N_nodes)
    edges = [(i, j) for (i,j) in zip(row, col)]
    g.add_edges(edges)
    g.es['weight'] = data

    return g

def _run_leiden(
    g,
    resolution,
):
    '''
    Run leiden clustering
    '''
    partition = la.find_partition(
        g, 
        la.CPMVertexPartition, 
        resolution_parameter = resolution
    )
    clusters = partition.membership
    return clusters

def _get_edge_chunks_random(edges_whole, num_chunks):
    '''
    Randomly split the edges into chunks
    '''
    edges_whole = edges_whole[torch.randperm(edges_whole.shape[0]), :]
    chunksize = edges_whole.shape[0] // num_chunks
    edges_whole_sections = torch.split(edges_whole, chunksize, dim = 0)
    return edges_whole_sections

def _get_edge_chunks_byTiling(edges_whole, num_chunks, x, y):
    '''
    split the edges into chunks by tiling the image (separate chunks by 2d coordinates here)
    '''
    src_x, src_y = x[edges_whole[:,0]], y[edges_whole[:,0]]
    dst_x, dst_y = x[edges_whole[:,1]], y[edges_whole[:,1]]
    centroid_x, centroid_y = (src_x + dst_x) / 2, (src_y + dst_y) / 2

    # get the tile size
    num_chunks_axis = np.round(np.sqrt(num_chunks)).astype(int)
    num_chunks = num_chunks_axis ** 2
    tile_size_x = (np.max(x) - np.min(x)) / num_chunks_axis
    tile_size_y = (np.max(y) - np.min(y)) / num_chunks_axis    

    for i in range(num_chunks_axis):
        for j in range(num_chunks_axis):
            min_x, max_x = np.min(x) + i * tile_size_x, np.min(x) + (i + 1) * tile_size_x
            min_y, max_y = np.min(y) + j * tile_size_y, np.min(y) + (j + 1) * tile_size_y
            tile_indices = np.where(
                (centroid_x >= min_x) & (centroid_x < max_x) & (centroid_y >= min_y) & (centroid_y < max_y)
            )[0]
            if i == 0 and j == 0:
                edges_whole_sections = [edges_whole[tile_indices]]
            else:
                edges_whole_sections.append(edges_whole[tile_indices])
    return edges_whole_sections, num_chunks

@torch.no_grad()
def run_leiden_predictedLink(
    bg: BrGraph,
    df_spots: pd.DataFrame,
    use_image: bool = True,
    pos_thresh: float = 0.6,
    neg_thresh: float = 0.5,
    resolution: float = 1e-2,
    num_edges_perSpot: int = 1000,
    n_neighbors: int = 10,
    num_iters: int = 10,
    split_edges_byTiling: bool = False,
):  
    # get all edges at once
    if bg.dimension == '2d':
        x, y = df_spots['x'].values, df_spots['y'].values
        coords = np.array([x, y]).T
    else:
        x, y, z = df_spots['x'].values, df_spots['y'].values, df_spots['z'].values
        coords = np.array([x, y, z]).T
    N_nodes = df_spots.shape[0]

    A = kneighbors_graph(coords, num_edges_perSpot, mode = 'connectivity', include_self = False)
    A = coo_matrix(A)
    row, col = A.row, A.col
    edges_whole = torch.from_numpy(np.array([row, col]).T).long()
    logger.info(f'Total number of edges for segmentation task is {edges_whole.shape[0]}')

    # chunk edges
    logger.info(f'Split edges into chunks, number of chunks: {num_iters}')
    if split_edges_byTiling == False:
        edges_whole_sections = _get_edge_chunks_random(edges_whole, num_iters)
    else:
        edges_whole_sections, num_iters = _get_edge_chunks_byTiling(edges_whole, num_iters, x, y)
    
    # prepare graph and node embeddings
    logger.info(f'Prepare graph and node embeddings')
    if (not hasattr(bg, 'graph_all')) and (not hasattr(bg, 'z_all')):
        graph_whole = BuildGraph_fromRaw(bg, df_spots.copy(), bg.features.copy(), n_neighbors = n_neighbors).cpu()
        pos_whole = graph_whole.pos
        bg.trainer_node.model.to('cpu')
        z_whole = bg.trainer_node.model.get_latent(graph_whole)
        del graph_whole
    else:
        if hasattr(bg, 'graph_all'):
            pos_whole = bg.graph_all.pos[bg.foreground_indices, :]
            del bg.graph_all
        else:
            pos_whole = bg.pos_all[bg.foreground_indices, :]
        z_whole = bg.z_all
        del bg.z_all

    # prepare image
    if (bg.image_raw is not None) and use_image:
        image_ = torch.from_numpy(bg.image_raw).double()
        image_ = image_[None, :, :, :].cuda()
    else:
        image_ = None

    # run for each chunk
    logger.info(f'Total number of iterations is {num_iters}')
    for iter in range(num_iters):
        logger.info(f'Iteration {iter + 1} of {num_iters} is running')
        edges_iter = edges_whole_sections[iter]

        if edges_iter.shape[0] > 0:
            edge_attr, edge_indices = _get_edge_embedding(bg.trainer_edge, pos_whole, image_, z_whole, edges_iter)
            pred_logits = _get_edge_labels(bg, edge_attr)
        else:
            edge_indices = torch.zeros((0, 2)).long() # empty tile
            pred_logits = np.array([])

        if iter == 0:
            edge_indices_whole = torch.clone(edge_indices)
            pred_logits_whole = pred_logits.copy()
        else:
            edge_indices_whole = torch.cat((edge_indices_whole, edge_indices), dim = 0)
            pred_logits_whole = np.concatenate((pred_logits_whole, pred_logits), axis = 0)

    # create adjancy matrix
    edge_indices_whole = edge_indices_whole.numpy()
    bg.edge_indices_whole = edge_indices_whole
    bg.pred_logits_whole = pred_logits_whole

    g = _create_adjacency_matrix(edge_indices_whole, pred_logits_whole, pos_thresh, neg_thresh, N_nodes)

    # clustering across resolutions
    clusters = {}
    if not isinstance(resolution, list):
        resolution = [resolution]
    for res in resolution:
        if res < 0:
            raise ValueError('resolution must be non-negative')
        else:
            clusters[res] = _run_leiden(g, res)

    return clusters