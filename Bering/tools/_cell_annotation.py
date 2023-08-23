import logging
import numpy as np
import pandas as pd

from anndata import AnnData
import scanpy as sc
from scipy.sparse import csr_matrix

from ..objects import Bering_Graph as BrGraph

def _ensemble_annotation(
    bg: BrGraph,
    min_transcripts: int = 50,
    min_dominant_nodes: int = 30,
    min_dominant_ratio: float = 0.6,
):
    '''
    annotate cells based on ensemble approach
    '''
    df_spots = bg.spots_all.copy()

    # get cell abundance 1
    df_abundance = pd.DataFrame(df_spots.groupby(['predicted_cells']).size())
    df_abundance.columns = ['n_cells']
    df_abundance = df_abundance.sort_values(by = ['n_cells'], ascending = False)

    cells_1 = np.unique(df_abundance.loc[df_abundance['n_cells'] > min_transcripts, :].index.values)

    # get cell abundance 2
    df_abundance = pd.DataFrame(df_spots.groupby(['predicted_cells', 'predicted_node_labels']).size())
    df_abundance.columns = ['n_cells']
    df_abundance = df_abundance.sort_values(by = ['n_cells'], ascending = False)
    df_abundance.reset_index(inplace = True)

    cells_2 = np.unique(df_abundance.loc[df_abundance['n_cells'] > min_dominant_nodes, 'predicted_cells'].values)

    # filter out qualified cells
    df_perc = (
        df_spots.groupby("predicted_cells")["predicted_node_labels"]\
            .apply(lambda x: x.value_counts(normalize=True).head(1))\
                .mul(100)\
                    .rename_axis(['predicted_cells','predicted_node_labels'])\
                        .reset_index(name='Perc')
    )
    cells_3 = df_perc.loc[(df_perc['predicted_node_labels'] != 'background')&(df_perc['Perc'] >= min_dominant_ratio), 'predicted_cells'].values

    qualified_cells = np.intersect1d(np.intersect1d(cells_1, cells_2), cells_3)

    df_perc.set_index('predicted_cells', inplace = True)
    qualified_labels = df_perc.loc[qualified_cells, 'predicted_node_labels'].values
    label_dict = dict(zip(qualified_cells, qualified_labels))

    all_labels = np.array(['Unknown'] * df_spots.shape[0], dtype = 'object')
    all_cells = np.array([-1] * df_spots.shape[0])
    predicted_cells = df_spots['predicted_cells'].values

    for cell, label in zip(qualified_cells, qualified_labels):
        indices = np.where(predicted_cells == cell)[0]
        all_labels[indices] = label
        all_cells[indices] = cell

    df_spots['ensembled_cells'] = all_cells
    df_spots['ensembled_labels'] = all_labels

    bg.spots_predicted = df_spots.copy()

    return df_spots, label_dict

def _create_anndata(
    df_spots_all: pd.DataFrame,
    label_dict: dict,
):
    '''
    Create anndata for segmented cells
    '''
    # ensembled annotations
    df_spots = df_spots_all.loc[df_spots_all['ensembled_cells'] != -1, :].copy()
    df_expr = df_spots.groupby(['ensembled_cells', 'features']).size().unstack(fill_value = 0)
    
    df_cells = pd.DataFrame(index = df_expr.index.values)
    df_cells['n_counts'] = np.sum(df_expr.values, axis = 1)
    df_cells['n_genes'] = np.sum(df_expr.values != 0, axis = 1)
    df_cells['cx'] = pd.DataFrame(df_spots.groupby(['ensembled_cells']).mean()['x']).loc[df_cells.index.values, 'x'].values
    df_cells['cy'] = pd.DataFrame(df_spots.groupby(['ensembled_cells']).mean()['y']).loc[df_cells.index.values, 'y'].values
    df_cells['predicted_labels'] = [label_dict[i] for i in df_cells.index.values]

    df_features = pd.DataFrame(index = df_expr.columns)
    df_features['n_cells'] = np.sum(df_expr.values != 0, axis = 0)

    adata_ensembl = AnnData(
        X = csr_matrix(df_expr.values),
        obs = df_cells,
        var = df_features,
    )

    # pure segmented annotations
    df_expr = df_spots_all.groupby(['predicted_cells', 'features']).size().unstack(fill_value = 0)
    
    df_cells = pd.DataFrame(index = df_expr.index.values)
    df_cells['n_counts'] = np.sum(df_expr.values, axis = 1)
    df_cells['n_genes'] = np.sum(df_expr.values != 0, axis = 1)
    df_cells['cx'] = pd.DataFrame(df_spots_all.groupby(['predicted_cells']).mean()['x']).loc[df_cells.index.values, 'x'].values
    df_cells['cy'] = pd.DataFrame(df_spots_all.groupby(['predicted_cells']).mean()['y']).loc[df_cells.index.values, 'y'].values

    df_features = pd.DataFrame(index = df_expr.columns)
    df_features['n_cells'] = np.sum(df_expr.values != 0, axis = 0)

    adata_segmented = AnnData(
        X = csr_matrix(df_expr.values),
        obs = df_cells,
        var = df_features,
    )

    return adata_ensembl, adata_segmented

def cell_annotation(
    bg: BrGraph,
    min_transcripts: int = 50,
    min_dominant_nodes: int = 30,
    min_dominant_ratio: float = 0.6,
):
    '''
    Annotate segmented cells based on ensemble strategy. A cell is considered as a qualified cell if it satisfies all of the following criteria:
    - Number of transcripts > min_transcripts;
    - Number of dominant nodes > min_dominant_nodes;
    - Ratio of dominant nodes > min_dominant_ratio.

    Other than that, the cell is not considered as a segmented cell and transcipts within this cell are assigned to the background.

    Parameters
    ----------
    bg: BrGraph
        Bering graph object
    min_transcripts: int
        Minimum number of transcripts for a cell to be considered
    min_dominant_nodes: int
        Minimum number of dominant nodes for a cell to be considered
    min_dominant_ratio: float
        Minimum ratio of dominant nodes for a cell to be considered
    
    '''
    df_spots, label_dict = _ensemble_annotation(
        bg = bg, 
        min_transcripts = min_transcripts,
        min_dominant_nodes = min_dominant_nodes, 
        min_dominant_ratio = min_dominant_ratio,
    )
    adata_ensembl, adata_segmented = _create_anndata(df_spots, label_dict)

    return df_spots, adata_ensembl, adata_segmented

def cell_analyze(
    adata: AnnData,
    min_counts: int = 10,
    target_sum: int = 1000,
    n_neighbors: int = 10,
):
    '''
    Run a standard analysis pipeline from single-cell data with raw counts. The pipeline includes:
    - Filter cells with low counts;
    - Normalize total counts;
    - Scale counts;
    - PCA;
    - Neighbor graph construction;
    - UMAP;
    - Leiden clustering.

    Parameters
    ----------
    adata: AnnData
        Anndata object. It can be the output of :func:`~.cell_annotation` function.
    min_counts: int
        Minimum number of counts for a gene to be considered
    target_sum: int
        Target sum of counts for normalization
    n_neighbors: int
        Number of neighbors for KNN graph construction

    Returns
    -------
    adata: AnnData
        Anndata object with updated fields
    '''
    sc.pp.filter_cells(adata, min_counts = min_counts)
    sc.pp.normalize_total(adata, target_sum = target_sum)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors = n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    return adata