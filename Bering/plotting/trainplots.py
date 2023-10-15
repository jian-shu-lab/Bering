import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import ConvexHull

import collections.abc as cabc
from types import MappingProxyType
from typing import Dict, Optional, Sequence, Union, Mapping, Any

from matplotlib.axes import Axes
from matplotlib import rcParams
from matplotlib import gridspec
import matplotlib.pyplot as plt 

def trainingplot(
    trainer,
    stages: Optional[Union[str, Sequence[str]]] = ['train', 'valid'],
    metrics: Optional[Union[str, Sequence[str]]] = None,
    color_train: str = 'blue',
    color_valid: str = 'orange',
    shapes_metrics: Optional[Union[str, Sequence[str]]] = None,
    same_axes_stages: bool = True,
    figsize: Optional[Sequence[float]] = None,
    savefig: bool = True,
    figname: Optional[str] = None,
):
    """Plot the training metrics of a trainer object.
    Args:
        trainer: Trainer object to plot the metrics for.
        metrics: Metrics to plot. If None, all available metrics are plotted.
            loss
            accuracy
            precision
            auc
    """
    if stages is None:
        stages = ['train', 'valid']
    elif isinstance(stages, str):
        stages = [stages]
    elif isinstance(stages, cabc.Sequence):
        stages = list(stages)

    if metrics is None:
        metrics = ['']
    elif isinstance(metrics, str):
        metrics = [metrics]
    elif isinstance(metrics, cabc.Sequence):
        metrics = list(metrics)

    epoches = list(trainer.epoch_record.values())[0]
    metrics_new = []
    metrics_values = []
    for stage in stages:
        for m in metrics:
            if m == 'loss':
                if stage == 'train':
                    values = list(trainer.train_loss_record.values())[0]
                elif stage == 'valid':
                    values = list(trainer.validation_loss_record.values())[0]
                else:
                    raise ValueError(f'Unknown stage: {stage}')
            elif m == 'accuracy':
                if stage == 'train':
                    values = list(trainer.acc_train_record.values())[0]
                elif stage == 'valid':
                    values = list(trainer.acc_test_record.values())[0]
                else:
                    raise ValueError(f'Unknown stage: {stage}')
            elif m == 'precision':
                if stage == 'train':
                    values = list(trainer.prec_train_record.values())[0]
                elif stage == 'valid':
                    values = list(trainer.prec_test_record.values())[0]
                else:
                    raise ValueError(f'Unknown stage: {stage}')
            elif m == 'auc':
                if stage == 'train':
                    values = list(trainer.auc_train_record.values())[0]
                elif stage == 'valid':
                    values = list(trainer.auc_test_record.values())[0]
                else:
                    raise ValueError(f'Unknown stage: {stage}')
            else:
                raise ValueError(f'Unknown metric: {m}')
            
            if len(values) == 0:
                warnings.warn(f'No values for metric {m} and stage {stage}.')
            else:
                if m not in metrics_new:
                    metrics_new.append(m)
                metrics_values.append(values)
    metrics = metrics_new

    # Plot the metrics
    n_from_stages = 1 if same_axes_stages else len(stages)
    n_from_metrics = len(metrics)
    num_subplots = n_from_stages * n_from_metrics
        
    if num_subplots == 1:
        fig, ax = plt.subplots(figsize=figsize)
        grid = None
    else:
        fig, grid = _panel_grid(num_subplots, figsize=figsize)

    if shapes_metrics is None:
        shapes_metrics = ['o', 's', 'v', '^', 'p', 'P', '*', 'h', 'H', 'D', 'd', '8', 'X']
    
    names_dict = {
        'loss': 'Loss',
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'auc': 'AUC'
    }
    stages_dict = {
        'train': 'Training',
        'valid': 'Validation'
    }

    # plot the metrics
    for i, stage in enumerate(stages):
        for j, metric in enumerate(metrics):
            if grid is not None:
                if same_axes_stages:
                    ax = plt.subplot(grid[j])
                else:
                    ax = plt.subplot(grid[len(metrics) * i + j])
            
            color = color_train if stage == 'train' else color_valid

            ax.plot(
                epoches, 
                metrics_values[i * len(metrics) + j], 
                color = color,
                marker = shapes_metrics[j],
                label = f'{stages_dict[stage]}'
            )
            ax.set_xlabel('Epoch')
            ax.set_ylabel(names_dict[metric])
            ax.set_title(f'{names_dict[metric]} vs. Epoch')
            
        if not same_axes_stages:
            h, l = ax.get_legend_handles_labels()
            ax.legend(*zip(*sorted(zip(h, l), key=lambda t: t[1])))

    if same_axes_stages:
        h, l = ax.get_legend_handles_labels()
        ax.legend(*zip(*sorted(zip(h, l), key=lambda t: t[1])))

    if savefig is None:
        return
    else:
        fig.savefig(figname, bbox_inches = 'tight', dpi = 300)

def _panel_grid(
    num_panels, 
    hspace = 0.35, 
    wspace = 0.30, 
    ncols = 4, 
    figsize = None, 
    **kwargs
):
    n_panels_x = min(ncols, num_panels)
    n_panels_y = np.ceil(num_panels / n_panels_x).astype(int)
    
    # each panel will have the size of rcParams['figure.figsize']
    figsize = rcParams['figure.figsize'] if figsize is None else figsize
    fig = plt.figure(
        figsize=(
            # n_panels_x * figsize[0] * (1 + wspace),
            n_panels_x * figsize[0],
            n_panels_y * figsize[1],
        ),
    )
    left = 0.2 / n_panels_x
    bottom = 0.13 / n_panels_y
    gs = gridspec.GridSpec(
        nrows=n_panels_y,
        ncols=n_panels_x,
        left=left,
        right=1 - (n_panels_x - 1) * left - 0.01 / n_panels_x,
        bottom=bottom,
        top=1 - (n_panels_y - 1) * bottom - 0.1 / n_panels_y,
        hspace=hspace,
        wspace=wspace,
        **kwargs,
    )
    return fig, gs