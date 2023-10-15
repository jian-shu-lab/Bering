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

if not os.path.isdir('plots'):
    os.mkdir('plots')

warnings.filterwarnings('ignore')

def scatterplot(
    spots_all: pd.DataFrame,
    color_by: Union[str, Sequence[str], None] = None,
    source_annotation: Union[str, Sequence[str], None] = 'raw',
    show_boundary: bool = True,
    min_points_boundary: int = 5,
    figsize: Optional[Sequence[float]] = None,
    s_bg: Optional[float] = None,
    s_fg: Optional[float] = None,
    boundary_color: str = 'black',
    color_gene: str = 'red',
    boundary_linewidth: float = 0.5,
    alpha_fg: float = 1.0,
    alpha_bg: float = 0.1,
    savename: Optional[str] = None,
    ax: Optional[Axes] = None,
    ncols: int = 2,
    hspace: float = 0.05,
    wspace: float = 0.0,
    frameon: bool = False,
    show_legend: bool = True,
    show_title: bool = True,
    panel_grid_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figure_settings_kwargs: Mapping[str, Any] = MappingProxyType({}),
    scatter_kwargs: Mapping[str, Any] = MappingProxyType({}),
    boundary_kwargs: Mapping[str, Any] = MappingProxyType({}),
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    save_kwargs: Mapping[str, Any] = MappingProxyType({}),
):    
    # plot grid spec
    if (
        not isinstance(color_by, str)
        and isinstance(color_by, cabc.Sequence)
        and len(color_by) > 1
    ) or (
        not isinstance(source_annotation, str)
        and isinstance(source_annotation, cabc.Sequence)
        and len(source_annotation) > 1
    ):
        if ax is not None:
            raise ValueError(
                'If passing multiple color_by values, '
                'ax must be None.'
            )
        
        if show_legend and wspace < 0.25:
            wspace += 0.25

        num_panels_from_color_by = (
            len(color_by) \
            if (not isinstance(color_by, str) \
                and isinstance(color_by, cabc.Sequence)) \
            else 1
        )
        num_panels_from_color_by = (
            len(source_annotation) \
            if (not isinstance(source_annotation, str) \
                and isinstance(source_annotation, cabc.Sequence)) \
            else 1
        )
        fig, grid = _panel_grid(
            hspace = hspace,
            wspace = wspace,
            ncols = ncols,
            num_panels = num_panels_from_color_by * num_panels_from_color_by,
            figsize = figsize,
            **panel_grid_kwargs,
        )
    else:
        grid = None 
        if ax is None:
            fig, ax = plt.subplots(
                figsize = figsize, 
                **figure_settings_kwargs,
            )
    
    # plot size and scatter size
    if figsize is None:
        figsize = rcParams['figure.figsize']

    dot_scale = 1.28e3
    if s_bg is None:
        s_bg = np.sqrt(
            (figsize[0] * figsize[1]) / dot_scale
        )
    if s_fg is None:
        s_fg = np.sqrt(
            (figsize[0] * figsize[1]) / dot_scale
        )

    # turn color_by into a list
    color_by_list = [color_by] if isinstance(color_by, str) else list(color_by)
    source_annotation_list = [source_annotation] \
        if isinstance(source_annotation, str) \
        else list(source_annotation)

    # plot for each subplot
    for i, source_annotation in enumerate(source_annotation_list):
        for j, color_by in enumerate(color_by_list):
            x, y, values, unique_groups, color_list = _get_values_groups(
                spots_all,
                color_by = color_by,
                source_annotation = source_annotation,
                color_gene = color_gene,
            )

            if grid:
                ax = plt.subplot(
                    grid[len(color_by_list) * i + j],
                    **figure_settings_kwargs,
                )

            if color_by == 'cell':
                np.random.shuffle(color_list)
            
            # keep color_list for label consistent
            if 'color_list_label' not in locals() and color_by == 'label':
                color_list_label = color_list

            if color_by == 'label':
                color_list = color_list_label

            for group, c in zip(unique_groups, color_list):
                xs, ys = x[values == group], y[values == group]
                if (color_by == 'cell' and group == 0) or \
                    (color_by == 'label' and group == 'background') or \
                    (color_by not in ['cell','label'] and group == 'others'):
                    color_ = '#DCDCDC'; alpha = alpha_bg
                    s = s_bg
                else:
                    alpha = alpha_fg
                    s = s_fg
                    color_ = c
                ax.scatter(
                    xs, ys, 
                    color = color_, 
                    s = s, 
                    label = group, 
                    alpha = alpha,
                    edgecolor = 'none',
                    **scatter_kwargs,
                )

            # add boundary
            if show_boundary:
                print(f'Plotting boundary of cells, progress bar ......')
                cells = spots_all[f'{source_annotation}_cells'].values
                for cell in tqdm(np.unique(cells)):
                    if cell == 0:
                        continue
                    
                    xc, yc = x[cells == cell], y[cells == cell]
                    if len(xc) < min_points_boundary:
                        continue
                    if len(np.unique(xc)) < 3:
                        continue
                    if len(np.unique(yc)) < 3:
                        continue
                        
                    try:
                        hull = ConvexHull(np.array([xc, yc]).T)
                    except:
                        continue
                    vertices = hull.vertices
                    vertices = np.concatenate(
                        [vertices, [vertices[0]]]
                    ) # ends meet

                    ax.plot(
                        xc[vertices], 
                        yc[vertices], 
                        color = boundary_color, 
                        linewidth = boundary_linewidth,
                        **boundary_kwargs,
                    )

            # add legend
            if color_by != 'cell':
                handles, _ = plt.gca().get_legend_handles_labels()
                # ncol = np.ceil(len(handles) / 8).astype(int)
                ncol_legend = 1
                ax.legend(
                    ncol = ncol_legend, 
                    markerscale = 10, 
                    fontsize = 6, 
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left', 
                    borderaxespad=0.,
                    **legend_kwargs,
                )
            if show_title:
                ax.set_title(
                    f'{color_by} ({source_annotation})', 
                )
            if not frameon:
                ax.axis('off')

    plt.tight_layout()

    if savename is None:
        fig.savefig(
            'test_scatterplot.png', 
            bbox_inches = 'tight', 
            dpi = 300,
            **save_kwargs,
        )
    else:
        fig.savefig(
            savename, 
            bbox_inches = 'tight', 
            dpi = 300,
            **save_kwargs,
        )


def _get_values_groups(
    spots_all: pd.DataFrame,
    color_by: Union[str, Sequence[str], None] = None,
    source_annotation: str = 'raw',
    color_gene: str = 'red',
):
    # get color key
    color_by_keys_dict = {
        ('cell', 'raw'): 'raw_cells',
        ('label', 'raw'): 'raw_labels',
        ('cell', 'predicted'): 'predicted_cells',
        ('label', 'predicted'): 'predicted_labels',
        ('cell', 'ensembled'): 'ensembled_cells',
        ('label', 'ensembled'): 'ensembled_labels',
    }

    genes_avail = spots_all.features.unique()

    if color_by in ['cell', 'label']:
        color_by_key = color_by_keys_dict[
            (color_by, source_annotation)
        ]
    elif color_by in genes_avail:
        color_by_key = 'features'
    else:
        raise ValueError(
            f'color_by must be \'cell\', \'label\' or a gene name'
        )

    # get values
    x, y = spots_all.x.values, spots_all.y.values
    values = spots_all[color_by_key].values

    if color_by not in ['cell', 'label']:
        values = np.array([color_by if v == color_by else 'others' for v in values])
    unique_groups = np.unique(values)

    # reorder groups
    if color_by == 'cell':
        unique_groups = np.concatenate(
            [[0], \
            np.setdiff1d(unique_groups, [0])]
        )
    elif color_by == 'label':
        unique_groups = np.concatenate(
            [['background'], \
            np.setdiff1d(unique_groups, ['background'])]
        )
    else:
        unique_groups = np.concatenate(
            [['others'], \
            np.setdiff1d(unique_groups, ['others'])]
        )

    # define color
    if color_by in ['cell', 'label']:
        color_list = plt.cm.get_cmap('tab20')(
            np.linspace(0, 1, len(unique_groups))
        )    
    else:
        color_list = ['#DCDCDC', color_gene]

    return x, y, values, unique_groups, color_list


def _panel_grid(hspace, wspace, ncols, num_panels, figsize, **kwargs):
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
