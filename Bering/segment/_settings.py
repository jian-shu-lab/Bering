import os
from typing import NamedTuple

class _SEGMENTATION_KEYS(NamedTuple):
    GRAPH_N_NEIGHBORS: int = 10
    NEIGHBOR_DISTANCE_BETA: float = 1.0
    LEIDEN_RESOLUTION: float = 1e-1
    CELL_MIN_TRANSCRIPTS: int = 30

SEGMENT_KEYS = _SEGMENTATION_KEYS()