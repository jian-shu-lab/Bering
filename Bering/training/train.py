import os
import logging
import datetime
import numpy as np
# from rich.progress import track
from tqdm import tqdm
from typing import Sequence

import torch
from torch_geometric.loader import DataLoader

from ._record import record_init, record, is_notebook
from ._trainer_node import TrainerNode
from ._trainer_edge import TrainerEdge
from ..models import GCN, BaselineMLP, EdgeClf
from ..objects import Bering_Graph as BrGraph

logger = logging.getLogger(__name__)

class EarlyStopper:
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
    def __init__(
        self, 
        patience: int = 6, 
        min_delta: float = 0.025,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        '''
        Check if the training should be stopped

        Parameters
        ----------
        validation_loss
            Validation loss value
        '''
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
    plot_ax_size: float = 5.0,
):
    '''
    Training procedure for node classifier
    '''
    record_init(trainer, trainer_type)
    early_stopper = EarlyStopper(early_stop_patience, early_stop_min_delta)
    
    pbar = tqdm(range(epoches), desc = 'Training node classifier', colour='blue')
    epoch_interval = 5
    plotting = False if is_notebook() else True
    
    for epoch in pbar:
        train_loss = trainer.update(train_loader) # loss function = CrossEntropy
        validation_loss = trainer.validate(test_loader)
        if early_stop and early_stopper.early_stop(validation_loss):
            break
        if (epoch % epoch_interval == 0) or epoch == (epoches - 1):
            if epoch == (epoches - 1):
                plotting = True
            record(
                trainer, 
                None,
                trainer_type, 
                epoch, 'node', train_loss, validation_loss, 
                train_loader, test_loader, logger,
                plot_folder = plot_folder, 
                plotting = plotting,
                ax_size = plot_ax_size,
                plot_name = f'{trainer_type}_node.png'
            )    
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
    plot_ax_size: float = 5.0,
):  
    '''
    Training procedure for edge classifier
    '''
    record_init(trainer, trainer_type)
    early_stopper = EarlyStopper(early_stop_patience, early_stop_min_delta)

    pbar = tqdm(range(epoches), desc = 'Training edge classifier', colour='red')
    epoch_interval = 5
    plotting = False if is_notebook() else True

    for epoch in pbar:
        train_loss = trainer.update(train_loader, image) # loss function = CrossEntropy
        validation_loss = trainer.validate(test_loader, image)
        if early_stop and early_stopper.early_stop(validation_loss):
            break
        if (epoch % epoch_interval == 0) or epoch == (epoches - 1):
            if epoch == (epoches - 1):
                plotting = True
            _, auc_test, _, prec_test, _, _ = record(
                trainer, 
                image,
                trainer_type, 
                epoch, 'edge', train_loss, validation_loss, 
                train_loader, test_loader, logger,
                plot_folder = plot_folder,
                plotting = plotting,
                plot_name = f'{trainer_type}_edge.png',
                ax_size = plot_ax_size,
            )
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
    plot_ax_size: float = 5.0,
    finetune: bool = False,
    baseline: bool = False,
):
    '''
    Training both node classification and edge classification models. 
    The training procedure is done by training node classifier :func:`~TrainerNode` first and then training edge classifier :func:`~TrainerEdge`.

    Parameters
    ----------
    bg
        Bering_Graph object
    node_gcnq_hidden_dims
        List of hidden layer dimensions for GCN in :func:`Bering.models.GCN`
    node_mlp_hidden_dims
        List of hidden layer dimensions for MLP in :func:`Bering.models.GCN`
    node_lr
        Learning rate for node classifier in :func:`TrainerNode`
    node_weight_decay
        Weight decay for node classifier in :func:`TrainerNode`
    node_foreground_weight
        Weight for segmented transcripts in loss function for node classifier in :func:`TrainerNode`
    node_background_weight
        Weight for background transcripts in loss function for node classifier in :func:`TrainerNode`
    node_epoches
        Number of epoches for node classifier
    node_early_stop
        Whether to use early stop for node classifier in :func:`~EarlyStopper`
    node_early_stop_patience
        Maximal number of consecutive times allowed to have
        loss greater than min_loss + min_delta before stopping for node classifier. See :func:`~EarlyStopper`
    node_early_stop_delta
        Minimal gap of a new loss and minimal loss for adding to a count to counter for node classifier. See :func:`~EarlyStopper`
    edge_distance_type
        Distance type for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_rbf_start
        Start of RBF kernel \mu for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_rbf_stop
        Stop of RBF kernel \mu for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_rbf_n_kernels
        Number of RBF kernels for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_rbf_learnable
        Whether to learn RBF kernels for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_image_conv2d_hidden_dims
        List of hidden layer dimensions for image encoder for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_image_mlp_hidden_dims
        List of hidden layer dimensions for image encoder for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_decoder_mlp_hidden_dims
        List of hidden layer dimensions for decoder for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_num_positive
        Number of positive edges for edge classifier. See :func:`~Bering.models.EdgeClf.forward`
    edge_num_negative
        Number of negative edges for edge classifier. See :func:`~Bering.models.EdgeClf.forward`
    edge_subimage_binsize
        Bin size for subimages for edge classifier. See :func:`~Bering.models.EdgeClf`
    edge_lr
        Learning rate for edge classifier. See :func:`TrainerEdge`
    edge_weight_decay
        Weight decay for edge classifier. See :func:`TrainerEdge`
    edge_epoches
        Number of epoches for edge classifier
    edge_early_stop
        Whether to use early stop for edge classifier. See :func:`~EarlyStopper`
    edge_early_stop_patience
        Maximal number of consecutive times allowed to have
        loss greater than min_loss + min_delta before stopping for edge classifier. See :func:`~EarlyStopper`
    edge_early_stop_delta
        Minimal gap of a new loss and minimal loss for adding to a count to counter for edge classifier. See :func:`~EarlyStopper`
    plot_ax_size
        Size of the plot for node and edge classifiers. See :func:`~record`
    finetune
        Whether to finetune the model. 
        - If ``True``, the model will be fine-tuned with pre-trained :func:`Bering.models.GCN` and :func::func:`Bering.models.EdgeClf`
        - If ``False``, the model will be trained from scratch with :func:`Bering.models.GCN` and `Bering.models.EdgeClf`
    baseline
        Whether to use baseline model. If ``True``, the model will be trained with :func:`Bering.models.BaselineMLP`. If ``False``, the model will be trained with :func:`Bering.models.GCN`.

    Returns
    -------
    ``bg.trainer_node``
        Node classifier: :func:`~TrainerNode`
    ``bg.trainer_edge``
        Edge classifier: :func:`~TrainerEdge`

    '''

    # initialize models
    if not finetune:
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
    if not finetune:
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
        plot_ax_size = plot_ax_size,
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

    if not finetune:
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
        plot_ax_size = plot_ax_size,
    )