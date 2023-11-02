import torch
import numpy as np
import pandas as pd
from typing import Optional, Sequence
from sklearn.neighbors import kneighbors_graph

from .._logger import LOGGING
logger = LOGGING()

def _get_cell_metadata(spots_seg):
    raw_cell_metadata = spots_seg.groupby(['raw_cells'])[['x','y','z']].median()
    gapDF = spots_seg.groupby(['raw_cells'])[['x','y','z']].agg(np.ptp)
    
    raw_cell_metadata.columns = ['cx', 'cy', 'cz'] # centroid of x, y, z
    gapDF.columns = ['dx', 'dy', 'dz'] # diameter of x,y,z
    
    raw_cell_metadata = pd.concat(
        [raw_cell_metadata, gapDF], axis = 1
    )
    raw_cell_metadata['d'] = [
        max(i,j,k) for (i,j,k) in zip(raw_cell_metadata.dx.values, raw_cell_metadata.dy.values, raw_cell_metadata.dz.values)
    ] 
    raw_cell_metadata = raw_cell_metadata.loc[(raw_cell_metadata['dx'] != 0) & (raw_cell_metadata['dy'] != 0), :].copy() # delete empty seg
    
    cell_meta = spots_seg.drop_duplicates(subset = ['raw_cells', 'raw_labels'])
    cell_meta.set_index('raw_cells', inplace = True)
    raw_cell_metadata['raw_labels'] = cell_meta.loc[raw_cell_metadata.index.values, 'raw_labels'].values
    return raw_cell_metadata

def _reindex_cell_ids(df_spots_seg):
    '''
    Index cell ids from 1 to n_cells (0 as the background)
    '''
    cell_ids = df_spots_seg['raw_cells'].unique()
    cell_ids_dict = dict(zip(cell_ids, np.arange(1, len(cell_ids) + 1)))

    df_spots_seg['raw_cells'] = df_spots_seg['raw_cells'].map(cell_ids_dict)
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
        required_features: Optional[Sequence[str]] = ['x', 'y', 'features', 'raw_cells', 'raw_labels'],
        dimension_3d: bool = False,
    ):
        logger.info('Building Bering Object ...')
        # store spots
        self.dimension = '3d' if dimension_3d else '2d'
        if self.dimension == '3d' and 'z' not in df_spots_seg.columns:
            raise ValueError("3D spots must contain 'z' coordinate")
        
        cols_unseg = [col for col in df_spots_unseg.columns if col not in ['raw_cells', 'raw_labels']]
        df_spots_unseg = df_spots_unseg.loc[:, cols_unseg].copy()

        self.spots_seg = df_spots_seg.copy()
        self.spots_unseg = df_spots_unseg.copy()

        if self.spots_seg.shape[0] != 0:
            self.spots_seg = _reindex_cell_ids(self.spots_seg) # reindex cell ids from 0 to n_cells

        self.spots_seg['raw_groups'] = 'foreground'
        self.spots_unseg['raw_cells'] = 0
        self.spots_unseg['raw_labels'] = 'background'
        self.spots_unseg['raw_groups'] = 'background'

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
            self.raw_cell_metadata = _get_cell_metadata(self.spots_seg)
            self.n_cells_raw = len(self.spots_seg.raw_cells.unique())
            self.raw_cells = self.spots_seg['raw_cells'].unique()
        else:
            self.raw_cell_metadata = pd.DataFrame()
            self.n_cells_raw = 0
            self.raw_cells = np.array([])

        # labels
        if self.spots_seg.shape[0] != 0:
            self.raw_labels = self.spots_all['raw_labels'].unique()
            self.raw_labels = np.append(
                np.setdiff1d(self.raw_labels, 'background'), 
                'background'
            ) # UNSEGMENTED as the last one
            self.n_labels_raw = len(self.raw_labels)
        else:
            self.raw_labels = np.array(['background'])
            self.n_labels_raw = 1
        
        self.labels_dict = dict(zip(self.raw_labels, range(self.n_labels_raw)))
        self.label_indices_dict = dict(zip(range(self.n_labels_raw), self.raw_labels))
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

        del self.spots_seg
        del self.spots_unseg
        # logging
        if self.dimension == '3d':
            logger.info(f'\tmin x: {self.XMIN}, min y: {self.YMIN}, max x: {self.XMAX}, max y: {self.YMAX}, min z: {self.ZMIN}, max z: {self.ZMAX}')
        else:
            logger.info(f'\tmin x: {self.XMIN}, min y: {self.YMIN}, max x: {self.XMAX}, max y: {self.YMAX}')
        logger.info(f'\tNumbers of segmented cell = {self.n_cells_raw}; labels = {self.n_labels_raw}; features = {self.n_features}')
        logger.info(f'\tLabel indices dictionary is {self.label_indices_dict}')
        for label in np.setdiff1d(self.raw_labels, ['background']):
            num_cells = len(np.where(self.raw_cell_metadata['raw_labels'].values == label)[0])
            logger.info(f'\tNumber of cells for {label}: {num_cells}')

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
        self.n_labels = bg2.n_labels_raw
        new_labels = np.setdiff1d(self.raw_labels, bg2.raw_labels)
        self.raw_labels = bg2.raw_labels
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