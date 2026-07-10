import numpy as np
import matplotlib.pyplot as plt
import caiman.utils.visualization
import caiman.base.rois


def plot_component_centers_and_contours(
    estimates, subset=None, background=None, thr=0.9, thr_method='nrg', figsize=(8, 8), ax=None
):
    """Plot each component's centroid and contour as a static sanity check,
    independent of the interactive Bokeh viewer.

    Populates estimates.coordinates (via caiman.utils.visualization.get_contours)
    if not already set; centers are always recomputed via caiman.base.rois.com.

    For 3D data (len(estimates.dims) == 3), each component's contour is drawn
    only at the z-slice closest to its own centroid, rather than overlaying
    every z-slice's contour -- overlaying all of them tends to look cluttered
    since components at different depths get drawn on top of each other.

    background: 2D (Y, X) array to plot components over; defaults to the max
        intensity projection (over Z, for 3D data) of the mean spatial
        footprint image.
    """
    if estimates.coordinates is None:
        estimates.coordinates = caiman.utils.visualization.get_contours(
            estimates.A, estimates.dims, thr=thr, thr_method=thr_method
        )

    is_3d = len(estimates.dims) == 3
    centers = caiman.base.rois.com(estimates.A, *estimates.dims)

    if background is None:
        background = np.array(estimates.A.mean(axis=1)).reshape(estimates.dims, order='F')
        if is_3d:
            background = background.max(axis=-1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(background, cmap='gray')
    if subset is not None:
        coordinates = [estimates.coordinates[i] for i in subset]
        centers = centers[subset]
    else:
        coordinates = estimates.coordinates

    for pars, center in zip(coordinates, centers):
        y, x = center[0], center[1]
        ax.plot(x, y, '+', color='red', markersize=8)
        ax.text(x, y, str(pars['neuron_id']), color='yellow', fontsize=8)

        contour = pars['coordinates']
        if is_3d:
            z = int(round(center[2]))
            z = min(max(z, 0), len(contour) - 1)
            contour = contour[z]
        ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=1)

    ax.set_title(f"{len(coordinates)} components")
    return ax
