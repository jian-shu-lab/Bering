from typing import NamedTuple

class _TRAIN_KEYS(NamedTuple):
    # LOSS_WEIGHTS_SEGMENTED: float = 2.0
    # LOSS_WEIGHTS_BACKGROUND: float = 0.5
    LOSS_WEIGHTS_SEGMENTED: float = 1.0
    LOSS_WEIGHTS_BACKGROUND: float = 1.0
    LOSS_WEIGHTS_POSEDGES: float = 1.0
    LOSS_WEIGHTS_NEGEDGES: float = 2.0
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 5e-4
    FOLDER_RECORD = 'figures/performance'

TRAIN_KEYS = _TRAIN_KEYS()