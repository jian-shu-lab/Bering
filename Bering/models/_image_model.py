import math
import logging
import collections
from typing import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    SPP layer (deprecated)
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    for i in range(len(out_pool_size)):        
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int(h_wid*out_pool_size[i] - previous_conv_size[0])
        w_pad = int(w_wid*out_pool_size[i] - previous_conv_size[1])
        new_previous_conv = nn.functional.pad(previous_conv, (0, w_pad, h_pad, 0))

        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(0, 0))
        x = maxpool(new_previous_conv)

        if(i == 0):
            spp = x.view(num_sample,-1)
        else:
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        
    return spp

class ImageEncoder(nn.Module):
    '''
    Convolutional neural network to learn representation from staining images of different sizes.

    Parameters
    ----------
    image_dims
        dimensions of the input image (n_samples x n_channels x W x H)
    cnn_layer_dims
        dimensions of CNN layers
    mlp_layer_dim
        dimensions of FC layers in the end
    spp_output_size
        size of spatial pyramid pooling. the total size of the spp layer in the dimension of input layer in FCN. Refer to :func:`~spatial_pyramid_pool`.
    '''
    def __init__(
        self, 
        image_dims: Sequence[int], 
        cnn_layer_dims: Sequence[int] = [6, 16, 32], 
        mlp_layer_dims: Sequence[int] = [32, 32],
        spp_output_size: Sequence[int] = [4, 2, 1],
    ):

        super().__init__()

        self.cnn_layer_dims = [image_dims[0]] + cnn_layer_dims
        self.mlp_layer_dims = [image_dims[1] * image_dims[2] * cnn_layer_dims[-1]] + mlp_layer_dims
        self.output_num = spp_output_size

        self.cnn_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer_CNN {i}",
                        nn.Sequential(
                            nn.Conv2d(
                                in_channel, out_channel, 
                                kernel_size=3, padding=1, stride=1,
                            ),
                            nn.ReLU(),
                        ),
                    )
                    for i, (in_channel, out_channel) in enumerate(
                        zip(self.cnn_layer_dims[:-1], self.cnn_layer_dims[1:])
                    )
                ]   
            )
        )
        self.fc1 = nn.Linear(cnn_layer_dims[-1] * sum(self.output_num) * 3, mlp_layer_dims[0])
        self.fc2 = nn.Linear(mlp_layer_dims[0], mlp_layer_dims[1])

    def get_conv2d_embedding(self, x: torch.Tensor):
        '''
        Get the shared convolution embedding (as the input of SPP) of a large image for edges that derived from this image.
        input image shape: 1 * n_channels * h * w

        Parameters
        ----------
        x
            input image tensor
        '''

        for i, layers in enumerate(self.cnn_layers):
            for layer in layers:
                x = layer(x)
        return x

    def forward(self, images: torch.Tensor):
        '''
        Run SPP layer and FC layers for edges with convolutional embedding features
        
        Parameters
        ----------
        images
            image embeddings (n_edges x n_conv2d_embeddings x W x H) for all edges as the input of SPP and FC layers
        '''

        num_sample = images.shape[0]
        for i in range(len(self.output_num)): 
            if (i == 0):
                spp = nn.AdaptiveMaxPool2d((self.output_num[i], self.output_num[i]))(images).view(num_sample, -1)
            else:
                spp = torch.cat((spp, nn.AdaptiveMaxPool2d((self.output_num[i], self.output_num[i]))(images).view(num_sample, -1)), 1)

        fc1 = self.fc1(spp)
        fc2 = self.fc2(fc1)

        return fc2