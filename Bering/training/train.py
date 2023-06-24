import os
import datetime
import logging
import numpy as np
from typing import Sequence

import torch
from torch_geometric.loader import DataLoader

from ._record import record_init, record
from ._trainer_node import TrainerNode
from ._trainer_edge import TrainerEdge
from ..models import GCN, BaselineMLP, EdgeClf
from ..objects import Bering_Graph as BrGraph

logger = logging.getLogger(__name__)

class EarlyStopper:
    def __init__(
        self, 
        patience: int = 6, 
        min_delta: float = 0.025,
    ):
        '''
        Early Stop implementation for loss values
        
        Paramemters
        -----------
        patience
            Maximal number of consecutive times allowed to have 
            loss greater than min_loss + min_delta before stopping
        min_delta
            Minimal gap of a new loss and minimal loss for adding to a count to counter
        '''
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def _trainNode(
    trainer: TrainerNode, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    trainer_type: str,
    plot_folder: str,
    epoches: int = 200, 
    early_stop: bool = True,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.025,
):
    '''
    Training procedure for node classifier
    '''
    record_init(trainer, trainer_type)
    early_stopper = EarlyStopper(early_stop_patience, early_stop_min_delta)
    for epoch in range(epoches):
        train_loss = trainer.update(train_loader) # loss function = CrossEntropy
        validation_loss = trainer.validate(test_loader)
        if early_stop and early_stopper.early_stop(validation_loss):
            break
        if epoch % 5 == 0:
            record(
                trainer, 
                None,
                trainer_type, 
                epoch, 'node', train_loss, validation_loss, 
                train_loader, test_loader, logger,
                plot_folder = plot_folder, 
                plot_name = f'{trainer_type}_node.png')    
    return trainer

def _trainEdge(
    trainer: TrainerEdge, 
    image,
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    trainer_type: str,
    plot_folder: str,
    epoches: int = 200, 
    early_stop: bool = True,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.025,
):  
    '''
    Training procedure for edge classifier
    '''
    record_init(trainer, trainer_type)
    early_stopper = EarlyStopper(early_stop_patience, early_stop_min_delta)
    for epoch in range(epoches):
        train_loss = trainer.update(train_loader, image) # loss function = CrossEntropy
        validation_loss = trainer.validate(test_loader, image)
        if early_stop and early_stopper.early_stop(validation_loss):
            break
        if epoch % 5 == 0:
            _, auc_test, _, prec_test, _, _ = record(
                trainer, 
                image,
                trainer_type, 
                epoch, 'edge', train_loss, validation_loss, 
                train_loader, test_loader, logger,
                plot_folder = plot_folder,
                plot_name = f'{trainer_type}_edge.png')
    return trainer


def Training(
    bg: BrGraph, 
    node_gcnq_hidden_dims: Sequence[int] = [256, 128, 64, 32, 16],
    node_mlp_hidden_dims: Sequence[int] = [16, 32, 32],
    node_lr: float = 1e-3,
    node_weight_decay: float = 5e-4,
    node_foreground_weight: float = 1.0,
    node_background_weight: float = 1.0,
    node_epoches: int = 50,
    node_early_stop: bool = False,
    node_early_stop_patience: int = 5,
    node_early_stop_delta: float = 0.05,
    edge_distance_type: str = 'rbf',
    edge_rbf_start: int = 0,
    edge_rbf_stop: int = 64,
    edge_rbf_n_kernels: int = 64, 
    edge_rbf_learnable: bool = True,
    edge_image_conv2d_hidden_dims: Sequence[int] = [6, 16, 32, 64, 128],
    edge_image_mlp_hidden_dims: Sequence[int] = [32, 64],
    edge_decoder_mlp_hidden_dims: Sequence[int] = [16, 8],
    edge_num_positive: int = 1000,
    edge_num_negative: int = 1000,
    edge_subimage_binsize: int = 5,
    edge_lr: float = 1e-3,
    edge_weight_decay: float = 5e-4,
    edge_epoches: int = 50,
    edge_early_stop: bool = False,
    edge_early_stop_patience: int = 5,
    edge_early_stop_delta: float = 0.05,
    retrain: bool = False,
    baseline: bool = False,
):
    '''
    Training both node classification and edge classification models
    '''
    # initialize models
    if not retrain:
        if not baseline:
            nodeclf = GCN(
                n_features = bg.n_node_features, 
                n_classes = bg.n_labels,
                gcn_hidden_layer_dims = node_gcnq_hidden_dims,
                mlp_hidden_layer_dims = node_mlp_hidden_dims,
            )
        else:
            nodeclf = BaselineMLP(
                n_features = bg.n_node_features,
                n_classes = bg.n_labels,
                mlp_hidden_layer_dims = node_gcnq_hidden_dims[:-1] + node_mlp_hidden_dims,
            )

        edgeclf = EdgeClf(
            n_node_latent_features = node_mlp_hidden_dims[1], 
            image = bg.image_raw,
            image_model = (bg.image_raw is not None),
            decoder_mlp_layer_dims = edge_decoder_mlp_hidden_dims,
            distance_type = edge_distance_type,
            rbf_start = edge_rbf_start,
            rbf_stop = edge_rbf_stop,
            rbf_n_kernels = edge_rbf_n_kernels,
            rbf_learnable = edge_rbf_learnable,
            encoder_image_layer_dims_conv2d = edge_image_conv2d_hidden_dims,
            encoder_image_layer_dims_mlp = edge_image_mlp_hidden_dims,
            subimage_binsize = edge_subimage_binsize,
            max_subimage_size = bg.window_size - edge_subimage_binsize,
            min_subimage_size = edge_subimage_binsize,
        )
    
    # train node clf
    performance_folder = 'figures/performance_' + datetime.datetime.now().strftime('%m-%d %H-%M-%S'); os.mkdir(performance_folder)
    if not retrain:
        bg.trainer_node = TrainerNode(nodeclf, lr = node_lr, weight_decay = node_weight_decay, weight_seg = node_foreground_weight, weight_bg = node_background_weight)

    keyword = 'training_GCN' if baseline == False else 'training_baseline'
    bg.trainer_node = _trainNode(
        bg.trainer_node, 
        bg.train_loader, 
        bg.test_loader, 
        keyword,
        performance_folder, 
        epoches = node_epoches,
        early_stop = node_early_stop,
        early_stop_patience = node_early_stop_patience,
        early_stop_min_delta = node_early_stop_delta,
    )

    # freeze node clf
    for param in bg.trainer_node.model.parameters():
        param.require_grad = False 
    
    # train edge clf
    if bg.image_raw is None:
        image_ = None
    else:
        image_ = torch.from_numpy(bg.image_raw).double().cuda()
        image_ = image_[None, :, :, :]

    if not retrain:
        bg.trainer_edge = TrainerEdge(
            edgeclf, bg.trainer_node.model, lr = edge_lr, weight_decay = edge_weight_decay, 
            num_pos_edges = edge_num_positive, num_neg_edges = edge_num_negative,
        )
    else:
        bg.trainer_edge.nodeclf_model = bg.trainer_node.model
    bg.trainer_edge = _trainEdge(
        bg.trainer_edge, 
        image_,
        bg.train_loader, 
        bg.test_loader, 
        keyword,
        performance_folder,
        epoches = edge_epoches,
        early_stop = edge_early_stop,
        early_stop_patience = edge_early_stop_patience,
        early_stop_min_delta = edge_early_stop_delta,
    )

# def Training_Baseline(
#     bg: BrGraph, 
#     epoches: int = 200,
#     mlp_hidden_dims: Sequence[int] = [256, 128, 64, 32, 16],
#     **kwargs,
# ):
#     '''
#     Training baseline MLP model for node classification
#     '''
#     # initialize
#     baseline = BaselineMLP(
#         n_features = bg.n_node_features, 
#         n_classes = bg.n_labels,
#         mlp_hidden_layer_dims = mlp_hidden_dims,
#     )
#     bg.trainer_base = TrainerNode(baseline, lr = 1e-3, weight_decay = 5e-4, **kwargs)
#     performance_folder = 'performance_' + datetime.datetime.now().strftime('%m-%d %H:%M:%S'); os.mkdir(performance_folder)

#     # train baseline model
#     bg.trainer_base = _trainNode(
#         bg.trainer_base, 
#         bg.train_loader, 
#         bg.test_loader, 
#         'training_Baseline',
#         performance_folder, 
#         epoches = 50,
#         early_stop = True,
#         early_stop_patience = 5,
#         early_stop_min_delta = 0.05,
#     )