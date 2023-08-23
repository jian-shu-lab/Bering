import logging
import numpy as np
        
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from ..models import GCN, EdgeClf
from ._settings import TRAIN_KEYS as TR_KEYS
logger = logging.getLogger(__name__)

class TrainerEdge(object):
    '''
    Trainer for edge classification model.

    Parameters
    ----------
    model
        Edge classification model
    nodeclf_model
        Node classification model. This model is used to get the latent representation from the node embeddings.
    num_pos_edges
        Number of positive edges
    num_neg_edges
        Number of negative edges
    lr
        Learning rate
    weight_decay
        Weight decay
    weight_posEdge
        Weight for positive edges in loss function
    weight_negEdge
        Weight for negative edges in loss function
    '''
    def __init__(
        self, 
        model: EdgeClf, 
        nodeclf_model: GCN,
        num_pos_edges: int,
        num_neg_edges: int,
        lr: float = TR_KEYS.LEARNING_RATE, 
        weight_decay: float = TR_KEYS.WEIGHT_DECAY,
        weight_posEdge: float = TR_KEYS.LOSS_WEIGHTS_POSEDGES,
        weight_negEdge: float = TR_KEYS.LOSS_WEIGHTS_NEGEDGES
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.nodeclf_model = nodeclf_model
        self.model.double()
        self.weight_posEdge = weight_posEdge
        self.weight_negEdge = weight_negEdge
        self.num_pos_edges = num_pos_edges
        self.num_neg_edges = num_neg_edges

    def update(
        self, 
        loader, 
        image: torch.Tensor,
    ):
        '''
        Update the model on the training set.

        Parameters
        ----------
        loader
            Training data loader: ``torch_geometric.data.DataLoader``
        image
            image as the input for the image encoder
        '''
        running_loss = 0.0
        for idx, batch_data in enumerate(loader):
            logger.info(f'Training batch {idx}')
            batch_data.to(self.device)
            self.model.train()
            self.optimizer.zero_grad()

            self.nodeclf_model.eval()
            z = self.nodeclf_model.get_latent(batch_data)
            if image is None:
                logits, edge_labels, _ = self.model(z, batch_data, self.num_pos_edges, self.num_neg_edges, None)
            else:    
                logits, edge_labels, _ = self.model(z, batch_data, self.num_pos_edges, self.num_neg_edges, torch.clone(image))
            
            loss = torch.nn.BCELoss()(logits.double(), edge_labels.double())
        
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss

    @torch.no_grad()
    def validate(
        self, 
        loader, 
        image: torch.Tensor,
    ):
        '''
        Validate the model on the test set.

        Parameters
        ----------
        loader
            Testing data loader: ``torch_geometric.data.DataLoader``
        image
            image as the input for the image encoder
        '''

        running_loss = 0.0
        for idx, batch_data in enumerate(loader):
            logger.info(f'Testing batch {idx}')
            batch_data.to(self.device)
            self.model.eval()

            self.nodeclf_model.eval()
    
            z = self.nodeclf_model.get_latent(batch_data)
            if image is None:
                logits, edge_labels, _ = self.model(z, batch_data, self.num_pos_edges, self.num_neg_edges, None)
            else:
                logits, edge_labels, _ = self.model(z, batch_data, self.num_pos_edges, self.num_neg_edges, torch.clone(image))

            # edge_weights = torch.FloatTensor([self.weight_negEdge] * len(edge_labels)).cuda()
            # edge_weights[torch.where(edge_labels == 1)[0]] = self.weight_posEdge
            # loss = torch.nn.BCELoss(weight = edge_weights)(logits.double(), edge_labels.double())

            loss = torch.nn.BCELoss()(logits.double(), edge_labels.double())
            running_loss += loss.item()

        return running_loss


    @torch.no_grad()
    def predict(self, batch_data, image):
        '''
        Predict the edge labels from the input data

        Parameters
        ----------
        batch_data
            Input data
        image
            image as the input for the image encoder
        '''
        
        self.model.eval()
        batch_data = batch_data.to(self.device)

        self.nodeclf_model.eval()
        z = self.nodeclf_model.get_latent(batch_data)
        if image is None:
            preds, y, _ = self.model(z, batch_data, self.num_pos_edges, self.num_neg_edges, None)
        else:
            preds, y, _ = self.model(z, batch_data, self.num_pos_edges, self.num_neg_edges, torch.clone(image))
        y, preds = y.detach().cpu().numpy(), preds.detach().cpu().numpy()

        preds_binary = np.round(preds).astype(np.int16)
        auc_score = roc_auc_score(y, preds)
        precision = average_precision_score(y, preds)
        
        accu = accuracy_score(y, preds_binary)
        err_pn = np.round(len(np.intersect1d(np.where(y == 1)[0], np.where(preds_binary == 0)[0])) / len(y), 2) # error rate of positive -> negative prediction
        err_np = np.round(len(np.intersect1d(np.where(y == 0)[0], np.where(preds_binary == 1)[0])) / len(y), 2)

        return auc_score, precision, accu, err_pn, err_np