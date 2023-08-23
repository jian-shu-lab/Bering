import os
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt


def _CREATE_CMAP(cmap_name, cmap_list):
    cmap_o = plt.get_cmap(cmap_name)
    for i in range(cmap_o.N):
        rgba = cmap_o(i)
        cmap_list.append(mpl.colors.rgb2hex(rgba))
    return cmap_list

def _PLOT_SETTINGS():
    if not os.path.isdir('figures'):
        os.mkdir('figures')

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams.update({'font.size': 18})
    plt.style.use('dark_background')

def _GET_CMAPS():
    cmap = []
    for name in ['Paired', 'tab20b', 'tab20c'] * 100:
        cmap = _CREATE_CMAP(name, cmap)
    return cmap

class _PLOTTING_KEYS(NamedTuple):
    # AX_WIDTH: float = 8.0
    # AX_HEIGHT: float = 8.0
    AX_WIDTH: float = 5.0
    AX_HEIGHT: float = 5.0
    AX_WIDTH_SLICE: float = 16.0
    AX_HEIGHT_SLICE: float = 16.0
    COLOR_SEG: str = '#FFFFFF'
    COLOR_BG: str = '#C0C0C0'
    SIZE_PT_LOCAL: float = 0.25
    SIZE_PT_CELL_ONSLICE: float = 2.5
    SIZE_PT_WINDOW: float = 6.0
    # SIZE_LEGEND_MARKER: float = 5.5
    # SIZE_LEGEND_FTSIZE: float = 14
    SIZE_LEGEND_MARKER: float = 4.0
    SIZE_LEGEND_FTSIZE: float = 10
    SIZE_FT_CELL: float = 10.0
    ALPHA_BG: float = 0.5
    WINDOW_LINEWIDTH: float = 1.0
    WINDOW_EDGECOLOR: str = 'white'
    FOLDER_NAME: str = 'figures'
    FILE_FORMAT: str = '.png'
    FILE_PREFIX_RAWSLICE: str = 'rawslice_'
    FILE_PREFIX_CLASSIFICATION: str = 'classification_'
    FILE_PREFIX_CLASSIFICATION_ZOOMEDIN: str = 'classification_zoomin_'
    FILE_PREFIX_SEGMENTATION: str = 'segmentation_'
    FILE_PREFIX_SEGMENTATION_POST: str = 'segmentation_post_'

PLOT_KEYS = _PLOTTING_KEYS()