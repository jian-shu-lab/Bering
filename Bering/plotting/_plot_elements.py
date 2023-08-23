import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from ._settings import _PLOT_SETTINGS, _GET_CMAPS
from ._settings import PLOT_KEYS as PLT_KEYS
from ..segment import SEGMENT_KEYS as SEG_KEYS

import warnings
warnings.filterwarnings("ignore")

_PLOT_SETTINGS()
CMAP = _GET_CMAPS() # discrete
CMAP_CONT = cm.get_cmap('binary_r') # continuous

def _show_legends(ax, markerscale, fontsize):
    h, l = ax.get_legend_handles_labels()
    ncol = 1 if len(l) < 8 else int(len(l)/8)
    ax.legend(h, l, markerscale = markerscale, fontsize = fontsize, ncol = ncol)
    return ax

def _raw_spots(img, x, y, ax, title = None, alpha = 0.7, **kwargs):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 0.5, cmap = 'gray')

    x = x - xmin 
    y = y - ymin

    ax.scatter(
        x, y, s = PLT_KEYS.SIZE_PT_LOCAL, color= PLT_KEYS.COLOR_SEG, alpha = alpha, **kwargs
    )
    ax.axis('off')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Raw Segmentation')
    return ax

def _raw_segmentation(img, x, y, ax, seg_types, **kwargs):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 0.5, cmap = '#DCDCDC')

    x = x - xmin 
    y = y - ymin

    for seg in np.unique(seg_types):
        xs = x[np.where(seg_types == seg)[0]]
        ys = y[np.where(seg_types == seg)[0]]

        if seg == 'segmented':
            ax.scatter(
                xs, ys, label = seg,
                s = PLT_KEYS.SIZE_PT_LOCAL, color= PLT_KEYS.COLOR_SEG, **kwargs
            )
        else:
            ax.scatter(
                xs, ys, label = seg,
                s = PLT_KEYS.SIZE_PT_LOCAL, color= PLT_KEYS.COLOR_BG, alpha = PLT_KEYS.ALPHA_BG, **kwargs
            )
    ax.axis('off')
    ax.set_title('Raw Segmentation')
    ax = _show_legends(ax, PLT_KEYS.SIZE_LEGEND_MARKER, PLT_KEYS.SIZE_LEGEND_FTSIZE)
    return ax

def _raw_cell_types(img, x, y, ax, raw_labels, label_to_col, s = None, **kwargs):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 0.5, cmap = 'gray')
    x = x - xmin 
    y = y - ymin

    for label in np.unique(raw_labels):
        xs = x[np.where(raw_labels == label)[0]]
        ys = y[np.where(raw_labels == label)[0]]

        s = PLT_KEYS.SIZE_PT_LOCAL if s is None else s
        if label == 'background':
            ax.scatter(
                xs, ys, label = label,
                s = s, color= label_to_col[label], alpha = PLT_KEYS.ALPHA_BG, **kwargs
            )
        else:
            ax.scatter(
                xs, ys, label = label,
                s = s, color= label_to_col[label], **kwargs
            )
    ax.axis('off')
    ax.set_title('Raw Annotations')
    ax = _show_legends(ax, PLT_KEYS.SIZE_LEGEND_MARKER, PLT_KEYS.SIZE_LEGEND_FTSIZE)
    return ax

def _raw_cell_types_addPatch(x, y, xw, yw, raw_labels, ax, label_to_col, **kwargs):
    ax = _raw_cell_types(x, y, ax, raw_labels, label_to_col, **kwargs)
    p = mpl.patches.Rectangle(
        xy = (np.min(xw), np.min(yw)), 
        width = (xw.max() - xw.min()), 
        height = (yw.max() - yw.min()),
        alpha = 1, facecolor = 'none', 
        edgecolor = PLT_KEYS.WINDOW_EDGECOLOR, linewidth = PLT_KEYS.WINDOW_LINEWIDTH
    )
    ax.add_patch(p)
    return ax

def _predicted_cell_types(img, x, y, ax, pred_labels, label_to_col, s = None, **kwargs):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 0.5, cmap = '#DCDCDC')
    x = x - xmin 
    y = y - ymin

    for label in np.unique(pred_labels):
        xs = x[np.where(pred_labels == label)[0]]
        ys = y[np.where(pred_labels == label)[0]]

        s = PLT_KEYS.SIZE_PT_LOCAL if s is None else s
        if label == 'background':
            ax.scatter(
                xs, ys, label = label,
                s = s, color= label_to_col[label], alpha = PLT_KEYS.ALPHA_BG, **kwargs
            )
        else:
            ax.scatter(
                xs, ys, label = label,
                s = s, color= label_to_col[label], **kwargs
            )
    ax.axis('off')
    ax.set_title('Predicted Annotations')
    ax = _show_legends(ax, PLT_KEYS.SIZE_LEGEND_MARKER, PLT_KEYS.SIZE_LEGEND_FTSIZE)
    return ax

def _predicted_probability(img, x, y, ax, probs, **kwargs):

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 0.5, cmap = 'gray')
    x = x - xmin 
    y = y - ymin

    ax_plot = ax.scatter(
        x, y, s = PLT_KEYS.SIZE_PT_LOCAL,
        c = probs, cmap = CMAP_CONT
    )
    ax.set_title('Predicted Probability')
    ax.axis('off')
    return ax, ax_plot

def _draw_cells(x, y, ax, cells, cell_source, **kwargs):
    c_index = 0
    avail_cells = np.setdiff1d(np.unique(cells), [''])
    for cell in avail_cells:
        xs = x[np.where(cells == cell)[0]]
        ys = y[np.where(cells == cell)[0]]

        if len(xs) < SEG_KEYS.CELL_MIN_TRANSCRIPTS:
            continue
        else:
            c_index += 1
            cx = np.mean(xs)
            cy = np.mean(ys)

            ax.scatter(
                xs, ys,
                s = PLT_KEYS.SIZE_PT_LOCAL, color= CMAP[c_index], **kwargs
            )
            ax.text(cx, cy, s = str(c_index), fontsize = PLT_KEYS.SIZE_FT_CELL)

    ax.axis('off')
    if cell_source == 'raw':
        ax.set_title('Raw Cells')
    elif cell_source == 'predicted':
        ax.set_title('Predicted Cells')

    return ax

def _draw_cells_withStaining(img, x, y, ax, cells, cell_source, **kwargs):
    # add image
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 1.0, cmap = 'gray')

    x = x - xmin 
    y = y - ymin
    c_index = 0

    avail_cells = np.unique(cells)
    for cell in avail_cells:
        xs = x[np.where(cells == cell)[0]]
        ys = y[np.where(cells == cell)[0]]

        if len(xs) < SEG_KEYS.CELL_MIN_TRANSCRIPTS:
            continue
        else:
            if cell != -1:
                c_index += 1
                cx = np.mean(xs)
                cy = np.mean(ys)

                ax.scatter(
                    xs, ys,
                    s = PLT_KEYS.SIZE_PT_LOCAL, color = CMAP[c_index], alpha = 1.0, **kwargs
                )
                ax.text(cx, cy, s = str(c_index), fontsize = PLT_KEYS.SIZE_FT_CELL)
            else:
                ax.scatter(
                    xs, ys,
                    s = PLT_KEYS.SIZE_PT_LOCAL, color = '#DCDCDC', alpha = 0.25, **kwargs
                )

    ax.axis('off')
    if cell_source == 'raw':
        ax.set_title('Raw Cells')
    elif cell_source == 'predicted':
        ax.set_title('Predicted Cells')
    else:
        ax.set_title(cell_source)

    return ax

def _draw_cells_withStaining_convexhull(img, x, y, ax, cells, labels, label_to_col, cell_source, **kwargs):
    # add image
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 1.0, cmap = 'gray')

    x = x - xmin 
    y = y - ymin
    
    # draw cell types
    for label in np.unique(labels):
        xs = x[np.where(labels == label)[0]]
        ys = y[np.where(labels == label)[0]]
        alpha = 1.0 if label not in  ['background', 'Unknown'] else 0.5
        ax.scatter(
            xs, ys,
            s = PLT_KEYS.SIZE_PT_LOCAL, color = label_to_col[label], alpha = alpha, **kwargs
        )
    
    # convex of cells
    from scipy.spatial import ConvexHull
    for cell in np.unique(cells):
        if cell == -1:
            continue
        xs = x[np.where(cells == cell)[0]]
        ys = y[np.where(cells == cell)[0]]
        if len(xs) < SEG_KEYS.CELL_MIN_TRANSCRIPTS:
            continue
        else:
            hull = ConvexHull(np.vstack([xs, ys]).T)
            for simplex in hull.simplices:
                ax.plot(xs[simplex], ys[simplex], color = 'black', linewidth = 0.5)

    ax.axis('off')
    if cell_source == 'raw':
        ax.set_title('Raw Cells')
    elif cell_source == 'predicted':
        ax.set_title('Predicted Cells')
    else:
        ax.set_title(cell_source)

    return ax

def _draw_cells_withStaining_withLinks(
    img, x, y, ax, cells, cell_source, 
    edges, pos_labels_ground, neg_labels_ground, pos_labels_pred, neg_labels_pred,
    num_edges_pertype = 40, color_pos_to_neg = '#FF6666', color_neg_to_pos = '#66B2FF',
    **kwargs
):
    # add image
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    if img is not None:
        img = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()
        ax.imshow(img, alpha = 0.7, cmap = 'gray')

    x = x - xmin 
    y = y - ymin
    c_index = 0
    if '' in np.unique(cells):
        avail_cells = np.setdiff1d(np.unique(cells), [''])
    else:
        avail_cells = np.unique(cells)
    for cell in avail_cells:
        xs = x[np.where(cells == cell)[0]]
        ys = y[np.where(cells == cell)[0]]

        if len(xs) < SEG_KEYS.CELL_MIN_TRANSCRIPTS:
            continue
        else:
            c_index += 1
            cx = np.mean(xs)
            cy = np.mean(ys)

            ax.scatter(
                xs, ys,
                s = PLT_KEYS.SIZE_PT_LOCAL, color= CMAP[c_index], **kwargs
            )
            ax.text(cx, cy, s = str(c_index), fontsize = PLT_KEYS.SIZE_FT_CELL)

    import random
    indices_pos_to_neg = np.intersect1d(pos_labels_ground, neg_labels_pred)
    indices_pos_to_neg = random.sample(indices_pos_to_neg.tolist(), min(len(indices_pos_to_neg), num_edges_pertype))

    indices_neg_to_pos = np.intersect1d(neg_labels_ground, pos_labels_pred)
    indices_neg_to_pos = random.sample(indices_neg_to_pos.tolist(), min(len(indices_neg_to_pos), num_edges_pertype))

    if len(indices_pos_to_neg) > 0:
        edges_pos_to_neg = edges[np.array(indices_pos_to_neg)]
        for edge in edges_pos_to_neg:
            srcx, srcy = x[edge[0]], y[edge[0]]
            dstx, dsty = x[edge[1]], y[edge[1]]
            ax.plot([srcx, dstx], [srcy, dsty], color = color_pos_to_neg, alpha = 0.5, linewidth = 0.5)

    if len(indices_neg_to_pos) > 0:
        edges_neg_to_pos = edges[np.array(indices_neg_to_pos)]
        for edge in edges_neg_to_pos:
            srcx, srcy = x[edge[0]], y[edge[0]]
            dstx, dsty = x[edge[1]], y[edge[1]]
            ax.plot([srcx, dstx], [srcy, dsty], color = color_neg_to_pos, alpha = 0.5, linewidth = 0.5)

    ax.axis('off')
    if cell_source == 'raw':
        ax.set_title('Raw Cells')
    elif cell_source == 'predicted':
        ax.set_title('Predicted Cells')
    else:
        ax.set_title(cell_source)

    return ax