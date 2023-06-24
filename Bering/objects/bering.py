import torch
import numpy as np
import pandas as pd
from typing import Optional, Sequence
from sklearn.neighbors import kneighbors_graph

from .._logger import LOGGING

logger = LOGGING()

def _get_cell_metadata(spots_seg):
    segmented = spots_seg.groupby(['segmented']).median()[['x','y']]
    gapDF = spots_seg.groupby(['segmented'])[['x','y']].agg(np.ptp)
    
    segmented.columns = ['cx', 'cy'] # centroid of x, y    
    gapDF.columns = ['dx', 'dy'] # diameter of x,y
    
    segmented = pd.concat(
        [segmented, gapDF], axis = 1
    )
    segmented['d'] = [
        max(i,j) for (i,j) in zip(segmented.dx.values, segmented.dy.values)
    ] 
    segmented = segmented.loc[(segmented['dx'] != 0) & (segmented['dy'] != 0), :].copy() # delete empty seg
    
    cell_meta = spots_seg.drop_duplicates(subset = ['segmented', 'labels'])
    cell_meta.set_index('segmented', inplace = True)
    segmented['labels'] = cell_meta.loc[segmented.index.values, 'labels'].values
    return segmented

def _reindex_cell_ids(df_spots_seg):
    '''
    Index cell ids from 0 to n_cells
    '''
    cell_ids = df_spots_seg['segmented'].unique()
    cell_ids_dict = dict(zip(cell_ids, np.arange(len(cell_ids))))

    df_spots_seg['segmented'] = df_spots_seg['segmented'].map(cell_ids_dict)
    return df_spots_seg

class Bering_Graph():
    '''
    Build Graph for training
        Input: dataframe contains 2D (3D) coordinates; Coarsely Segmented Cell IDs; Transcript (Protein) IDs
        Input column names: (index), x, y, (z), features, segmented, labels (df_spots_seg)
        Input column names: (index), x, y, (z), features (df_spots_unseg)
        Make sure indices in "df_spot_seg" and "df_spots_unseg" do not overlap. Currently we only support indices as "integer" type
        image is a concatenated microscopy image contains ndims layers (e.g. dapi + cytoplasm + membrane)
    '''
    def __init__(
        self, 
        df_spots_seg: pd.DataFrame, 
        df_spots_unseg: pd.DataFrame,
        image: np.ndarray,
        channels: Optional[Sequence[str]] = None,
        use_features: Optional[Sequence[str]] = None,
        required_features: Optional[Sequence[str]] = ['x', 'y', 'features', 'segmented', 'labels'],
    ):
        # store spots
        self.spots_seg = df_spots_seg.copy()
        self.spots_unseg = df_spots_unseg.copy()

        if self.spots_seg.shape[0] != 0:
            self.spots_seg = _reindex_cell_ids(self.spots_seg) # reindex cell ids from 0 to n_cells

        self.spots_seg['groups'] = 'segmented'
        self.spots_unseg['segmented'] = -1
        self.spots_unseg['groups'] = 'unsegmented'
        self.spots_unseg['labels'] = 'background'

        self.spots_all = pd.concat(
            [self.spots_seg, self.spots_unseg], axis = 0
        )
        self.spots_all['molecule_id'] = np.arange(self.spots_all.shape[0]) # reindex molecule ids from 0 to n_molecules
        self.spots_all.set_index('molecule_id', inplace = True)

        for feature in required_features: # check features
            if feature not in self.spots_all.columns:
                raise ValueError(f"Required feature {feature} not found in the input dataframe")

        self.transcripts_ids = self.spots_all.index.values
        self.n_transcripts = len(self.transcripts_ids)

        # registration of segmented samples
        if self.spots_seg.shape[0] != 0:
            self.segmented = _get_cell_metadata(self.spots_seg)
            self.n_segmented = len(self.spots_seg.segmented.unique())
            self.samples = self.spots_seg['segmented'].unique()
        else:
            self.segmented = pd.DataFrame()
            self.n_segmented = 0
            self.samples = np.array([])

        # labels
        if self.spots_seg.shape[0] != 0:
            self.labels = self.spots_all['labels'].unique()
            self.labels = np.append(
                np.setdiff1d(self.labels, 'background'), 
                'background'
            ) # UNSEGMENTED as the last one
            self.n_labels = len(self.labels)
        else:
            self.labels = np.array(['background'])
            self.n_labels = 1
        
        self.labels_dict = dict(zip(self.labels, range(self.n_labels)))
        self.label_indices_dict = dict(zip(range(self.n_labels), self.labels))
        self.label_to_col = {}

        # registration of features
        if use_features is None:
            self.features = pd.DataFrame(self.spots_all.groupby('features').size())
        else:
            _spots_all = self.spots_all.copy()
            _spots_all['features'] = pd.Categorical(_spots_all['features'], categories = use_features)
            self.features = pd.DataFrame(_spots_all.groupby('features').size())
            del _spots_all
        self.features.columns = ['counts']
        self.n_features = self.features.shape[0]

        # shape
        self.XMIN, self.XMAX = np.min(self.spots_all.x.values), np.max(self.spots_all.x.values)
        self.YMIN, self.YMAX = np.min(self.spots_all.y.values), np.max(self.spots_all.y.values)

        # morphogical design
        x = self.spots_all.x.values
        y = self.spots_all.y.values
        coords = np.array([x, y]).T
        A = kneighbors_graph(coords, n_neighbors = 10, mode = 'distance')
        self.percentile_dist = np.percentile(A.data, 20) # 20th percentile closest distance in knn graph
        self.epsilon = 1e-6
        self.n_morphology = 1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.performance = {}

        # image features
        self.image = image
        if self.image is None:
            self.image_raw = self.image_normalize = None
            self.image_ndims = 0
            self.channels = None 
            self.n_channels = 0
        else:
            self.image_raw = self.image.copy()
            self.image_ndims = self.image.shape[0]
            self.channels = channels
            self.n_channels = len(channels)

        # logging
        logger.info(f'min x: {self.XMIN}, min y: {self.YMIN}, max x: {self.XMAX}, max y: {self.YMAX}')
        logger.info(f'Numbers of segmented cell = {self.n_segmented}; labels = {self.n_labels}; features = {self.n_features}')
        logger.info(f'Label indices dictionary is {self.label_indices_dict}')
        for label in np.setdiff1d(self.labels, ['background']):
            num_cells = len(np.where(self.segmented['labels'].values == label)[0])
            logger.info(f'Number of cells for {label}: {num_cells}')