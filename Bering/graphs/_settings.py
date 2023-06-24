from typing import NamedTuple

class _GRAPHS_KEYS(NamedTuple):
    WINDOW_WIDTH: float = 10.0
    WINDOW_HEIGHT: float = 10.0
    WINDOW_MIN_POINTS: int = 20
    N_NEIGHBORS: int = 10
    BATCH_SIZE: int = 16
    TRAINING_RATIO: float = 0.8
    
GRAPH_KEYS = _GRAPHS_KEYS()