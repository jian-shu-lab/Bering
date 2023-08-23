import torch
import numpy as np
import pandas as pd
from typing import Optional, Sequence
from sklearn.neighbors import kneighbors_graph

from .._logger import LOGGING

logger = LOGGING()

def _get_cell_metadata(spots_seg):
    segmented = spots_seg.groupby(['segmented']).median()[['x','y','z']]
    gapDF = spots_seg.groupby(['segmented'])[['x','y','z']].agg(np.ptp)
    
    segmented.columns = ['cx', 'cy', 'cz'] # centroid of x, y, z
    gapDF.columns = ['dx', 'dy', 'dz'] # diameter of x,y,z
    
    segmented = pd.concat(
        [segmented, gapDF], axis = 1
    )
    segmented['d'] = [
        max(i,j,k) for (i,j,k) in zip(segmented.dx.values, segmented.dy.values, segmented.dz.values)
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

def _normalize(image):
    for channel_id in range(image.shape[0]):
        image_channel = image[channel_id, :]
        image_channel = image_channel / 255
        image[channel_id,:] = image_channel
    return image    

class Bering_Graph():
    '''
    Build Bering Object for training and prediction purposes. The input data contains segmented spots and unsegmented spots.    
    In addition, image and the channel description are required for image-dependent training .

    Parameters
    ----------
    df_spots_seg
        Dataframe for segmented spots, which contains 2D/3D coordinates ("x", "y", "z"); Transcript /Protein IDs ("features"); 
        Coarsely Segmented Cell IDs ("segmented"); and labels of cells ("labels")
    df_spots_unseg
        Dataframe for unsegmented spots, which contains 2D/3D coordinates ("x", "y", "z"); Transcript /Protein IDs ("features").
    image
        Concatenated microscopy image contains ndims layers (e.g. dapi + cytoplasm + membrane). The shape is (n_channels, height, width).
    channels
        Channel names of the image
    use_features
        Features to use
    required_features
        Required features in the input dataframe
    dimension_3d
        Whether the spots are 3D or not

    Returns
    -------
    Bering_Graph
    '''

    def __init__(
        self, 
        df_spots_seg: pd.DataFrame, 
        df_spots_unseg: pd.DataFrame,
        image: np.ndarray = None,
        channels: Optional[Sequence[str]] = None,
        use_features: Optional[Sequence[str]] = None,
        required_features: Optional[Sequence[str]] = ['x', 'y', 'features', 'segmented', 'labels'],
        dimension_3d: bool = False,
    ):
        # store spots
        self.dimension = '3d' if dimension_3d else '2d'
        if self.dimension == '3d' and 'z' not in df_spots_seg.columns:
            raise ValueError("3D spots must contain 'z' coordinate")
        
        cols_unseg = [col for col in df_spots_unseg.columns if col not in ['segmented', 'labels']]
        df_spots_unseg = df_spots_unseg.loc[:, cols_unseg].copy()

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
        if self.dimension == '3d':
            self.ZMIN, self.ZMAX = np.min(self.spots_all.z.values), np.max(self.spots_all.z.values)

        # morphogical design
        x = self.spots_all.x.values
        y = self.spots_all.y.values
        if self.dimension == '3d':
            z = self.spots_all.z.values
            coords = np.array([x, y, z]).T
        else:
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
            self.add_image_features(normalize=True)

        # logging
        if self.dimension == '3d':
            logger.info(f'min x: {self.XMIN}, min y: {self.YMIN}, max x: {self.XMAX}, max y: {self.YMAX}, min z: {self.ZMIN}, max z: {self.ZMAX}')
        else:
            logger.info(f'min x: {self.XMIN}, min y: {self.YMIN}, max x: {self.XMAX}, max y: {self.YMAX}')
        logger.info(f'Numbers of segmented cell = {self.n_segmented}; labels = {self.n_labels}; features = {self.n_features}')
        logger.info(f'Label indices dictionary is {self.label_indices_dict}')
        for label in np.setdiff1d(self.labels, ['background']):
            num_cells = len(np.where(self.segmented['labels'].values == label)[0])
            logger.info(f'Number of cells for {label}: {num_cells}')

    def use_settings(self, bg2):
        '''
        Borrow settings from another Bering_Graph object to ensure same training settings.

        Parameters
        ----------
        bg2
            Another Bering_Graph object
        '''
        self.n_node_features = bg2.n_node_features
        self.window_size = bg2.window_size

        self.features = bg2.features
        self.labels_dict = bg2.labels_dict
        self.label_indices_dict = bg2.label_indices_dict
        self.n_labels = bg2.n_labels
        new_labels = np.setdiff1d(self.labels, bg2.labels)
        self.labels = bg2.labels
        self.label_to_col = bg2.label_to_col
        for new_label in new_labels:
            self.label_to_col[new_label] = np.random.rand(3)

        if hasattr(bg2, 'trainer_node'):
            self.trainer_node = bg2.trainer_node
        if hasattr(bg2, 'trainer_edge'):
            self.trainer_edge = bg2.trainer_edge

    def add_image_features(
        self,
        normalize: bool = True,
    ):
        '''
        Add image features to spots.

        Parameters
        ----------
        normalize
            Normalize image to [0, 1]
        '''
        img = self.image 

        if normalize:
            img = _normalize(img)
            self.image_normalize = img.copy()

        channels = self.channels
        spots = self.spots_all.copy()
        xr = np.floor(spots.x.values).astype(np.int16)
        yr = np.floor(spots.y.values).astype(np.int16)

        if self.dimension == '3d':
            zr = np.floor(spots.z.values).astype(np.int16)
            for channel_id, channel in enumerate(channels):
                img_channel = img[channel_id, :, :, :]
                spots[channel] = img_channel[zr, yr, xr]
        else:
            for channel_id, channel in enumerate(channels):
                img_channel = img[channel_id, :, :]
                spots[channel] = img_channel[yr, xr]
        
        self.spots_all = spots.copy()
        self.spots_seg = spots.loc[spots['groups'] == 'segmented', :].copy()
        self.spots_unseg = spots.loc[spots['groups'] == 'unsegmented', :].copy()