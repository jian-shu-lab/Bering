import logging
import numpy as np
from typing import Sequence, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import MLP

from ._image_model import ImageEncoder
from ._utils import GaussianSmearing

logger = logging.getLogger(__name__)
from ._edge_process import _sample_edges, _binning_coordinates

def _get_image_graph(
    pos: torch.Tensor,
    image: torch.Tensor,
    src_coords: torch.Tensor,
    dst_coords: torch.Tensor,
    conv2d_padding: int = 10,
):
    '''
    Extract the graph image for edge embedding; recalculate source and destination node coordinates
    '''

    pos_graph = pos[:,[1,2]]
    xmin, xmax = int(torch.min(pos_graph[:,0]))-conv2d_padding, int(torch.max(pos_graph[:,0]))+conv2d_padding
    ymin, ymax = int(torch.min(pos_graph[:,1]))-conv2d_padding, int(torch.max(pos_graph[:,1]))+conv2d_padding

    xmin = max(0, xmin); xmax = min(xmax, int(image.shape[-1]))
    ymin = max(0, ymin); ymax = min(ymax, int(image.shape[-2]))

    image_graph = image[:,:,ymin:ymax,xmin:xmax]

    src_coords[:,0] = src_coords[:,0] - xmin; src_coords[:,1] = src_coords[:,1] - ymin
    dst_coords[:,0] = dst_coords[:,0] - xmin; dst_coords[:,1] = dst_coords[:,1] - ymin 

    src_coords = torch.round(src_coords).long()
    dst_coords = torch.round(dst_coords).long()
    
    return image_graph, src_coords, dst_coords

def _get_binned_coordinates(
    src_coords: torch.Tensor,
    dst_coords: torch.Tensor,
    image_binsize: int,
    min_image_size: int,
    max_image_size: int,
):
    minx, maxx, dist_x_appro = _binning_coordinates(
        src_coords[:,0], dst_coords[:,0], image_binsize = image_binsize, min_image_size = min_image_size, max_image_size = max_image_size, 
    )
    miny, maxy, dist_y_appro = _binning_coordinates(
        src_coords[:,1], dst_coords[:,1], image_binsize = image_binsize, min_image_size = min_image_size, max_image_size = max_image_size, 
    )
    dist_bins_2d = torch.concat((dist_x_appro[:,None], dist_y_appro[:,None]), axis = 1)
    avail_bins = torch.unique(dist_bins_2d, dim = 0)
    return minx, maxx, miny, maxy, avail_bins, dist_bins_2d

class EdgeClf(nn.Module):
    '''
    Edge classifier model which learns node classification embedding, image embedding and distance kernel

    Parameters
    ----------
    n_node_latent_features
        Number of latent features from node classification model
    image
        Image tensor for computing the conv2d embedding
    image_model
        Whether to use image model
    decoder_mlp_layer_dims
        List of hidden layer dimensions for MLP
    distance_type
        Type of RBF distance kernel. Options are None, 'positional', 'rbf'
    rbf_start
        Start of RBF kernel parameter \mu. Refer to :func:`~GaussianSmearing`
    rbf_stop
        Stop of RBF kernel parameter \mu. Refer to :func:`~GaussianSmearing`
    rbf_n_kernels
        Number of kernels in RBF kernel. Refer to :func:`~GaussianSmearing`
    rbf_learnable
        Whether to learn the RBF kernel in backpropagation. Refer to :func:`~GaussianSmearing`
    encoder_image_layer_dims_conv2d
        List of hidden layer dimensions for CNN in image encoder. Refer to :func:`~ImageEncoder`
    encoder_image_layer_dims_mlp
        List of hidden layer dimensions for MLP in image encoder. Refer to :func:`~ImageEncoder`
    subimage_binsize
        Binning size of subimage of edges
    max_subimage_size
        Maximal size of subimage of edges after crop
    min_subimage_size
        Minimal size of subimage of edges after crop

    '''
    def __init__(
        self, 
        n_node_latent_features: int, 
        image: Union[torch.Tensor, np.ndarray],
        image_model: bool = True,
        decoder_mlp_layer_dims: Sequence[int] = [16, 8],
        distance_type: Optional[str] = 'rbf',
        rbf_start: float = 0,
        rbf_stop: float = 64,
        rbf_n_kernels: int = 64,
        rbf_learnable: bool = True,
        encoder_image_layer_dims_conv2d: Sequence[int] = [6, 16, 32, 64, 128],
        encoder_image_layer_dims_mlp: Sequence[int] = [32, 64],
        subimage_binsize: int = 5,
        max_subimage_size: int = 40,
        min_subimage_size: int = 5,
    ):
        super().__init__()

        # RBF distance kernel
        self.distance_type = distance_type
        if self.distance_type == 'rbf':
            self.rbf_learnable = rbf_learnable
            self.rbf_kernel = GaussianSmearing(
                start = rbf_start, stop = rbf_stop, 
                num_kernel = rbf_n_kernels, centered=False, learnable = rbf_learnable,
            )
        
        # image encoder
        if (image is not None) and image_model:
            self.image_model = True
            self.image_binsize = subimage_binsize
            self.max_image_size = max_subimage_size
            self.min_image_size = min_subimage_size
            
            self.encoder_image = ImageEncoder(image_dims = image.shape, cnn_layer_dims = encoder_image_layer_dims_conv2d, mlp_layer_dims = encoder_image_layer_dims_mlp)
            self.n_image_features = encoder_image_layer_dims_mlp[-1]
            num_parameters_image = sum([p.numel() for p in self.encoder_image.parameters() if p.requires_grad])
        else:
            self.image_model = False
            self.encoder_image = None
            num_parameters_image = 0

        # n latent embeddings
        n_latent_features_ = n_node_latent_features * 2
        if distance_type is None:
            n_latent_features_ += 0
        if distance_type == 'positional':
            n_latent_features_ += 2
        elif distance_type == 'rbf':
            n_latent_features_ += rbf_n_kernels

        if self.encoder_image is not None:
            n_latent_features_ += encoder_image_layer_dims_mlp[-1]

        # FC decoder
        self.decoder = MLP(
            [n_latent_features_] + list(decoder_mlp_layer_dims) + [1],
            act = 'relu',
            norm = 'batch_norm'
        )

        # parameters
        if self.distance_type == 'rbf':
            num_parameters_rbf = sum([p.numel() for p in self.rbf_kernel.parameters() if p.requires_grad])
        num_parameters_decoder = sum([p.numel() for p in self.decoder.parameters() if p.requires_grad])

        logger.info(f'Number of CNN parameters is {num_parameters_image}')
        if self.distance_type == 'rbf':
            logger.info(f'Number of RBF kernel parameters is {num_parameters_rbf}')
        logger.info(f'Number of MLP decoder parameters is {num_parameters_decoder}')

    def forward(
        self, 
        z_node: torch.Tensor, 
        data: torch.Tensor, 
        num_pos_edges: int, 
        num_neg_edges: int, 
        image: torch.Tensor, 
        conv2d_padding: int = 10,
    ):
        '''
        Run the decoder model from latent space z.
        Before running the decoder, random positive and negative edges are generated as the input. 

        Parameters
        -----------
        z
            Latent features from pretrained node classification (n samples x n latent features)
        data
            Input data loader (several graphs)
        num_pos_edges
            Number of positive edges
        num_neg_edges
            Number of negative edges
        image
            Image tensor for computing the conv2d embedding
        conv2d_padding
            add paddings in the conv2d embedding 
        '''

        # sample random edges each time
        edge_index, edge_labels, edge_graph_indices = _sample_edges(data, num_pos_edges, num_neg_edges)
        
        for idx, graph_index in enumerate(torch.unique(edge_graph_indices)):
            
            # get src / dst indices
            src = edge_index[0, torch.where(edge_graph_indices == graph_index)[0]]
            dst = edge_index[1, torch.where(edge_graph_indices == graph_index)[0]]
            edge_labels_graph = edge_labels[torch.where(edge_graph_indices == graph_index)[0]]

            # get weights
            weights = data.pos[src, -1] * data.pos[dst, -1]

            # get attributes
            edge_attr = torch.cat([z_node[src], z_node[dst]], dim = -1)
            # src_coords = data.pos[src, :][:,[1,2]] # 2d
            # dst_coords = data.pos[dst, :][:,[1,2]]
            src_coords = data.pos[src, :][:,[1,2,3]] # 3d
            dst_coords = data.pos[dst, :][:,[1,2,3]]

            if self.distance_type == 'rbf':
                edge_attr_rbf = self.rbf_kernel(x = src_coords, y = dst_coords)            
                edge_attr = torch.cat((edge_attr, edge_attr_rbf), axis = -1)

            if self.image_model:
                import time
                # get conv2d embeddings
                t0 = time.time()
                pos_graph = data.pos[data.ptr[graph_index]:data.ptr[graph_index+1], :]
                image_graph, src_coords, dst_coords = _get_image_graph(pos_graph, image, src_coords, dst_coords, conv2d_padding)
                image_graph = self.encoder_image.get_conv2d_embedding(image_graph)
                t1 = time.time()
                logger.info(f'---Get image graph time: {(t1-t0):.5f} s')
                
                # binning coordinates
                minx, maxx, miny, maxy, avail_bins, dist_bins_2d = _get_binned_coordinates(src_coords, dst_coords, self.image_binsize, self.min_image_size, self.max_image_size)
                t2 = time.time()
                logger.info(f'---Get all binned coordinates time: {(t2-t1):.5f} s')

                # run the model for eachedge
                edge_attr_image = torch.empty((src_coords.shape[0], self.n_image_features)).double().cuda()
                for avail_bin in avail_bins:
                    t3 = time.time()
                    bin_indices = torch.where((dist_bins_2d == avail_bin).all(dim=1))[0]

                    subimages = []
                    for i,j in enumerate(bin_indices):
                        subimage = image_graph[:,:,miny[j]:maxy[j], minx[j]:maxx[j]]
                        subimages.append(subimage)

                    subimages = torch.cat(subimages, axis = 0)
                    t4 = time.time()
                    logger.info(f'--------bin size: {avail_bin}, num of subimages: {len(bin_indices)}')
                    logger.info(f'--------Concat / read time: {(t4-t3):.5f} s')
                    edge_attr_image_bin = self.encoder_image(subimages)
                    edge_attr_image[bin_indices, :] = edge_attr_image_bin
                    t5 = time.time()
                    logger.info(f'--------encode subimages time: {(t5-t4):.5f} s')

                edge_attr = torch.cat([edge_attr, edge_attr_image], dim = -1)

            if idx == 0:
                edge_attr_combined = edge_attr
                edge_labels_combined = edge_labels_graph
            else:
                edge_attr_combined = torch.cat([edge_attr_combined, edge_attr], dim = 0)
                edge_labels_combined = torch.cat([edge_labels_combined, edge_labels_graph])

        pred_labels = self.decoder(edge_attr_combined)
        pred_labels = F.sigmoid(pred_labels).squeeze()

        return pred_labels, edge_labels_combined, weights
