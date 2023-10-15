import os

from ._settings import PLOT_KEYS, _GET_CMAPS, _PLOTTING_KEYS
from .plot import Plot_SliceImages, Plot_Classification, Plot_Segmentation, Plot_Spots
from .plot_post import Plot_Segmentation_Post, Plot_Classification_Post
from ._plot_elements import _draw_cells_withStaining
from .scatterplots import scatterplot
from .trainplots import trainingplot

FOLDER = PLOT_KEYS.FOLDER_NAME
if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)