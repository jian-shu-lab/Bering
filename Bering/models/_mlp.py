import collections
from typing import Sequence

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, BatchNorm

class BaselineMLP(nn.Module):
    '''
    Node classification Model with Multilayer Perceptron (MLP) without using graph models.

    Parameters
    ----------
    n_features
        Number of input features
    n_classes
        Number of predicted classes
    mlp_hidden_layer_dims
        List of hidden layer dimensions for MLP
    '''
    
    def __init__(
        self, 
        n_features, 
        n_classes,
        mlp_hidden_layer_dims: Sequence[int] = [256, 128, 64, 32, 16],
        # num_mlp_layers_remain: int = 2,
    ):
        super().__init__()
        # self.num_mlp_layers_remain = num_mlp_layers_remain
        self.mlp_layer_dims = [n_features] + list(mlp_hidden_layer_dims) + [n_classes]
        
        # self.mlp = MLP(
        #     self.mlp_layers, 
        #     act = 'relu',
        #     norm = 'batch_norm'
        # )

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

    def forward(self, data):
        '''
        Forward pass

        Parameters
        ----------
        data
            Pytorch Geometric Data object
        '''

        x = data.x
        for i, layers in enumerate(self.mlp_layers):
            for layer in layers:
                x = layer(x)
        return x

    @torch.no_grad()
    def get_latent(self, data, num_mlp_layers_remain = 2):
        '''
        Get latent features

        Parameters
        ----------
        data
            Pytorch Geometric Data object
        num_mlp_layers_remain
            The number of layers from the last MLP layer. The output of this layer is the latent features.
        '''
        x = data.x
        num_mlp_layers = len(self.mlp_layers) - num_mlp_layers_remain

        for i, layers in enumerate(self.mlp_layers):
            if i < num_mlp_layers:
                for layer in layers:
                    x = layer(x)
        return x    