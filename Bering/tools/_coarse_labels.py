import numpy as np
import pandas as pd

def _label_cells_by_markers(
    bg, 
    marker_table,
    top_n_cells = 3,
):
    1