import numpy as np
import matplotlib.pyplot as plt
import caiman.utils.visualization
import caiman.base.rois


def plot_component_centers_and_contours(
    estimates, subset=None, view_axis=2, background=None, thr=0.9, thr_method='nrg', figsize=(8, 8), ax=None
):
    """Plot each component's centroid and contour as a static sanity check,
    independent of the interactive Bokeh viewer.

    For 3D data (len(estimates.dims) == 3), the movie is projected/sliced
    along `view_axis` (an index into estimates.dims): 2 (default) gives the
    usual top-down view (Z collapsed, Y vs X); use 0 or 1 for a side view
    (Y or X collapsed) to check whether components that overlap in the top
    view are actually separated along depth rather than being duplicates of
    the same thing. Each component's contour is drawn only at the slice
    (along view_axis) closest to its own centroid, rather than overlaying
    every slice's contour, which tends to look cluttered.

    Contours for view_axis=2 are cached on estimates.coordinates (and reused
    across calls); other view_axis values are always recomputed fresh and
    left uncached, so they don't clobber the default top-view cache.

    background: 2D array to plot components over; defaults to the max
        intensity projection (along view_axis, for 3D data) of the mean
        spatial footprint image.
    """
    is_3d = len(estimates.dims) == 3
    centers = caiman.base.rois.com(estimates.A, *estimates.dims)

    if is_3d:
        row_dim, col_dim = (d for d in range(3) if d != view_axis)
        if view_axis == 2 and estimates.coordinates is not None:
            coordinates = estimates.coordinates
        else:
            coordinates = caiman.utils.visualization.get_contours(
                estimates.A, estimates.dims, thr=thr, thr_method=thr_method, slice_dim=view_axis
            )
            if view_axis == 2:
                estimates.coordinates = coordinates
    else:
        if estimates.coordinates is None:
            estimates.coordinates = caiman.utils.visualization.get_contours(
                estimates.A, estimates.dims, thr=thr, thr_method=thr_method
            )
        coordinates = estimates.coordinates

    if background is None:
        background = np.array(estimates.A.mean(axis=1)).reshape(estimates.dims, order='F')
        if is_3d:
            background = background.max(axis=view_axis)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.imshow(background, cmap='gray')
    if subset is not None:
        coordinates = [coordinates[i] for i in subset]
        centers = centers[subset]

    for pars, center in zip(coordinates, centers):
        if is_3d:
            plot_x, plot_y = center[col_dim], center[row_dim]
        else:
            plot_y, plot_x = center[0], center[1]
        ax.plot(plot_x, plot_y, '+', color='red', markersize=8)
        ax.text(plot_x, plot_y, str(pars['neuron_id']), color='yellow', fontsize=8)

        contour = pars['coordinates']
        if is_3d:
            s = int(round(center[view_axis]))
            s = min(max(s, 0), len(contour) - 1)
            contour = contour[s]
        ax.plot(contour[:, 0], contour[:, 1], color='red', linewidth=1)

    ax.set_title(f"{len(coordinates)} components")
    return ax


def plot_component_views(estimates, subset=None, view_axes=(2, 0), background=None,
                          thr=0.9, thr_method='nrg', figsize=(6, 6)):
    """Plot the same components from multiple projection angles side by
    side (default: top view collapsing Z, and a side view collapsing Y), to
    check whether components that overlap in one view are actually
    separated along the collapsed axis rather than being duplicates.

    See plot_component_centers_and_contours for what each view_axis means.
    """
    axis_labels = {0: 'side view (Y collapsed)', 1: 'side view (X collapsed)', 2: 'top view (Z collapsed)'}
    fig, axes = plt.subplots(1, len(view_axes), figsize=(figsize[0] * len(view_axes), figsize[1]))
    axes = np.atleast_1d(axes)
    for ax, view_axis in zip(axes, view_axes):
        plot_component_centers_and_contours(
            estimates, subset=subset, view_axis=view_axis, background=background,
            thr=thr, thr_method=thr_method, ax=ax
        )
        ax.set_title(axis_labels.get(view_axis, f'view_axis={view_axis}'))
    fig.tight_layout()
    return fig, axes
