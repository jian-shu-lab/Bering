from typing import Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from ..models import GCN, BaselineMLP
from ._settings import TRAIN_KEYS as TR_KEYS

class TrainerNode(object):
    '''
    Trainer for node classification model.

    Parameters
    ----------
    model
        Node classification model
    lr
        Learning rate
    weight_decay
        Weight decay
    weight_seg
        Weight for segmented transcripts in loss function
    weight_bg
        Weight for background transcripts in loss function

    '''
    def __init__(
        self, 
        model: Optional[nn.Module] = GCN,
        lr: float = TR_KEYS.LEARNING_RATE, 
        weight_decay: float = TR_KEYS.WEIGHT_DECAY,
        weight_seg: float = TR_KEYS.LOSS_WEIGHTS_SEGMENTED,
        weight_bg: float = TR_KEYS.LOSS_WEIGHTS_BACKGROUND
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.double()
        self.weight_seg = weight_seg
        self.weight_bg = weight_bg

    def update(
        self, 
        loader, 
    ):
        '''
        Update the model on the training set.

        Parameters
        ----------
        loader
            Training data loader: ``torch_geometric.data.DataLoader``
        '''

        running_loss = 0.0
        for batch_data in loader:
            n_classes = int(batch_data.y.shape[1])

            weights = torch.FloatTensor([self.weight_seg] * (n_classes - 1) + [self.weight_bg]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight = weights)

            batch_data.to(self.device)
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model(batch_data)
            loss = criterion(logits, batch_data.y.float())
        
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss

    @torch.no_grad()
    def validate(
        self, 
        loader, 
    ):
        '''
        Validate the model on the test set.

        Parameters
        ----------
        loader
            Validation data loader: ``torch_geometric.data.DataLoader``
        '''
        running_loss = 0.0
        self.model.eval()

        for batch_data in loader:
            n_classes = int(batch_data.y.shape[1])

            weights = torch.FloatTensor([self.weight_seg] * (n_classes - 1) + [self.weight_bg]).to(self.device)
            criterion = nn.CrossEntropyLoss(weight = weights)

            batch_data.to(self.device)

            logits = self.model(batch_data)
            loss = criterion(logits, batch_data.y.float())
        
            running_loss += loss.item()

        return running_loss


    @torch.no_grad()
    def predict(self, batch_data, device = None):
        '''
        Predict the class probabilities of the input data.

        Parameters
        ----------
        batch_data
            Input data: ``torch_geometric.data.Data``
        device
            Device to run the model. Options: 'cuda' or 'cpu'
        '''
        self.model.eval()
        if device is None:
            batch_data = batch_data.to(self.device)
        else:
            batch_data = batch_data.to(device)
            self.model.to(device)

        logits = self.model(batch_data)
        logits = torch.softmax(logits, dim=-1).detach() # nsamples x nclasses

        return logits