import collections
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.nn import Linear
from torch_geometric.nn import GCNConv, BatchNorm, MLP

class GCN(nn.Module):
    '''
    Node classification Model with Graph Convolutional Networks (GCN) and Multilayer Perceptron (MLP).

    Parameters
    ----------
    n_features
        Number of input features
    n_classes
        Number of predicted classes
    gcn_hidden_layer_dims
        List of hidden layer dimensions for GCN
    mlp_hidden_layer_dims
        List of hidden layer dimensions for MLP
    dropout_rate
        Dropout rate
    '''
    def __init__(
        self, 
        n_features: int, 
        n_classes: int, 
        gcn_hidden_layer_dims: Sequence[int] = [256, 128, 64, 32, 16],
        mlp_hidden_layer_dims: Sequence[int] = [16, 32, 32],
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.gcn_layer_dims = [n_features] + list(gcn_hidden_layer_dims)
        self.mlp_layer_dims = list(mlp_hidden_layer_dims) + [n_classes]
        
        self.gcn_n_layers = len(self.gcn_layer_dims)
        self.mlp_n_layers = len(self.mlp_layer_dims)

        self.gcn_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer_GCN {i}",
                        nn.Sequential(
                            GCNConv(n_in, n_out),
                            BatchNorm(n_out),
                            nn.ReLU(),
                            nn.Dropout(p = dropout_rate)
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(self.gcn_layer_dims[:-1], self.gcn_layer_dims[1:])
                    )
                ]   
            )
        )

        self.mlp_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer_FC {i}",
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            BatchNorm(n_out),
                            nn.ReLU(),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(self.mlp_layer_dims[:-1], self.mlp_layer_dims[1:])
                    )
                ]   
            )
        )

        # self.mlp = MLP(
        #     self.mlp_layer_dims, 
        #     act = 'relu',
        #     norm = 'batch_norm'
        # )

    def forward(self, data):
        '''
        Get the prediction of the model from the input data
        '''
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for i, layers in enumerate(self.gcn_layers):
            for layer in layers:
                if isinstance(layer, GCNConv):
                    x = layer(x, edge_index, edge_weight)
                else:
                    x = layer(x)

        for i, layers in enumerate(self.mlp_layers):
            for layer in layers:
                if isinstance(layer, Linear):
                    x = layer(x)
                else:
                    x = layer(x)

        # x = self.mlp(x)
        return x

    @torch.no_grad()
    def get_latent(self, data, num_mlp_layers = 1):
        '''
        Get the latent representation of the model from the input data

        Parameters
        ----------
        data
            Input data
        num_mlp_layers
            Number of MLP layers to use to get the latent representation
        '''
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, layers in enumerate(self.gcn_layers):
            for layer in layers:
                if isinstance(layer, GCNConv):
                    x = layer(x, edge_index, edge_weight)
                else:
                    x = layer(x)
        
        for i, layers in enumerate(self.mlp_layers):
            if i < num_mlp_layers:
                for layer in layers:
                    if isinstance(layer, Linear):
                        x = layer(x)
                    else:
                        x = layer(x)
        return x