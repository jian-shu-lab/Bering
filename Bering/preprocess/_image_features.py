import cv2
import numpy as np

from skimage import feature, filters, measure, segmentation
from sympy import Segment

from . import _utils as utils
from ..objects import Bering_Graph as BrGraph
from ._image_preprocessing import _normalize, _scale

def add_image_features(
    bg: BrGraph,
    normalize: bool = True,
):
    img = bg.image 

    if normalize:
        img = _normalize(img)
        bg.image_normalize = img.copy()

    channels = bg.channels
    spots = bg.spots_all.copy()
    xr = np.floor(spots.x.values).astype(np.int16)
    yr = np.floor(spots.y.values).astype(np.int16)

    for channel_id, channel in enumerate(channels):
        img_channel = img[channel_id, :, :]
        spots[channel] = img_channel[yr, xr]
    
    bg.spots_all = spots.copy()
    bg.spots_seg = spots.loc[spots['groups'] == 'segmented', :].copy()
    bg.spots_unseg = spots.loc[spots['groups'] == 'unsegmented', :].copy()