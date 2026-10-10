import numpy as np
import matplotlib.pyplot as plt
import caiman.utils.visualization
import caiman.base.rois
import os




def plot_component_centers_and_contours(
    estimates, movie=None, subset=None, view_axis=2, background=None, thr=0.9, thr_method='nrg', figsize=(8, 8), ax=None
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

    movie: raw/motion-corrected movie array, shape (T,) + estimates.dims.
        Required (unless `background` is given explicitly): the background
        is always the movie's max intensity projection over time (and,
        for 3D data, further max-projected along view_axis) -- not derived
        from the component footprints, which would mostly be black outside
        detected components.
    background: precomputed 2D array to plot components over instead of
        deriving one from `movie`.
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
        if movie is None:
            raise ValueError(
                "plot_component_centers_and_contours needs either `movie` (to compute the max "
                "intensity projection) or an explicit `background`."
            )
        movie = np.asarray(movie)
        expected_shape = (movie.shape[0],) + tuple(estimates.dims)
        if movie.shape != expected_shape:
            raise ValueError(f"Expected movie shape {expected_shape}, got {movie.shape}.")
        background = movie.mean(axis=0)
        if is_3d:
            background = background.mean(axis=view_axis)

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


def plot_response_heatmap(estimates, movie, trials, component_idx, condition, fr,
                           condition_cols=('stim', 'concentration'), baseline_seconds=5, response_seconds=15,
                           peak_trace='F_dff', view_axis=2, cmap='jet', percent=True, thr=0.9, thr_method='nrg',
                           vmin=None, vmax=None, vpercentile=(1, 99), show_all_contours=False, ax=None):
    """Plot a per-pixel response heatmap (%dF/F, or raw dF if percent=False)
    for one stimulus condition, with the given component's contour overlaid
    -- similar to a published false-color glomerulus response map.

    For every trial matching `condition` (a tuple of values for
    condition_cols, e.g. ('CO2', '0.50%')): finds the frame of peak response
    within response_seconds after onset, using component_idx's own
    `peak_trace` (e.g. estimates.F_dff[component_idx]) to locate the peak;
    averages the 3 raw movie frames surrounding that peak as the "response"
    image; averages baseline_seconds of frames before onset as "baseline";
    takes their difference (as a % of baseline if percent=True). Averages
    that across the condition's repetitions. This is computed directly from
    the raw/motion-corrected movie, not from the component spatial
    footprints, which would only show detected-ROI weights rather than the
    actual measured response. For 3D data, the result is taken at the
    single z-slice (along view_axis) nearest component_idx's own centroid,
    not averaged across every z-plane -- matching the same slice used to
    draw that component's contour in plot_component_centers_and_contours,
    so the image isn't diluted by depths the component isn't even in.

    vmin/vmax: color scale bounds. If both left as None (default), they're
    set from vpercentile of the response image instead of its full min/max
    -- the full range is usually stretched by a handful of noisy/outlier
    pixels, which mutes the color contrast (saturation) on everything else.
    Pass explicit vmin/vmax (e.g. matching a fixed scale across figures) to
    override this.

    show_all_contours: if True, also draw every other component's contour
        lightly (thin, gray) for anatomical context, alongside
        component_idx's (bold, white) -- mirrors how published glomerulus
        maps outline the whole array of ROIs, not just one.
    """
    condition_cols = list(condition_cols)
    movie = np.asarray(movie)
    expected_shape = (movie.shape[0],) + tuple(estimates.dims)
    if movie.shape != expected_shape:
        raise ValueError(f"Expected movie shape {expected_shape}, got {movie.shape}.")

    conditions = list(trials[condition_cols].itertuples(index=False, name=None))
    mask = np.array([c == tuple(condition) for c in conditions])
    matching = trials[mask]
    if matching.empty:
        raise ValueError(f"No trials match condition {tuple(condition)} for columns {condition_cols}.")

    baseline_frames = int(round(baseline_seconds * fr))
    response_frames = int(round(response_seconds * fr))
    trace = getattr(estimates, peak_trace)[component_idx]

    deltas = []
    for onset in matching['onset_frame'].to_numpy():
        onset = int(round(onset))
        search_end = min(onset + response_frames, len(trace))
        peak_frame = onset + int(np.argmax(trace[onset:search_end]))
        peak_start = max(0, peak_frame - 1)
        peak_end = min(movie.shape[0], peak_frame + 2)  # 3 frames centered on the peak

        baseline = movie[max(0, onset - baseline_frames):onset].mean(axis=0)
        response = movie[peak_start:peak_end].mean(axis=0)
        deltas.append(100 * (response - baseline) / (baseline + 1e-6) if percent else response - baseline)
    delta_f = np.mean(deltas, axis=0)

    is_3d = len(estimates.dims) == 3
    centers = caiman.base.rois.com(estimates.A, *estimates.dims)
    if is_3d:
        z_slice = int(round(centers[component_idx][view_axis]))
        z_slice = min(max(z_slice, 0), estimates.dims[view_axis] - 1)
        delta_f = delta_f.take(z_slice, axis=view_axis)
        coordinates = caiman.utils.visualization.get_contours(
            estimates.A, estimates.dims, thr=thr, thr_method=thr_method, slice_dim=view_axis
        )
    else:
        coordinates = caiman.utils.visualization.get_contours(estimates.A, estimates.dims, thr=thr, thr_method=thr_method)

    if vmin is None and vmax is None:
        vmin, vmax = np.percentile(delta_f, vpercentile)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(delta_f, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='% dF/F' if percent else 'dF')

    to_draw = range(len(coordinates)) if show_all_contours else [component_idx]
    for i in to_draw:
        contour = coordinates[i]['coordinates']
        if is_3d:
            s = int(round(centers[i][view_axis]))
            s = min(max(s, 0), len(contour) - 1)
            contour = contour[s]
        is_target = (i == component_idx)
        ax.plot(contour[:, 0], contour[:, 1], color='black' if is_target else 'gray',
                 linewidth=2 if is_target else 0.8, alpha=1.0 if is_target else 0.6)

    ax.set_title(f"Component {component_idx}: {' '.join(str(v) for v in condition)} response")
    return ax


def plot_component_views(estimates, movie=None, subset=None, view_axes=(2, 0), background=None,
                          thr=0.9, thr_method='nrg', figsize=(6, 6)):
    """Plot the same components from multiple projection angles side by
    side (default: top view collapsing Z, and a side view collapsing Y), to
    check whether components that overlap in one view are actually
    separated along the collapsed axis rather than being duplicates.

    See plot_component_centers_and_contours for what each view_axis means
    and why `movie` is needed for the background.
    """
    axis_labels = {0: 'side view (Y collapsed)', 1: 'side view (X collapsed)', 2: 'top view (Z collapsed)'}
    fig, axes = plt.subplots(1, len(view_axes), figsize=(figsize[0] * len(view_axes), figsize[1]))
    axes = np.atleast_1d(axes)
    for ax, view_axis in zip(axes, view_axes):
        plot_component_centers_and_contours(
            estimates, movie=movie, subset=subset, view_axis=view_axis, background=background,
            thr=thr, thr_method=thr_method, ax=ax
        )
        ax.set_title(axis_labels.get(view_axis, f'view_axis={view_axis}'))
    fig.tight_layout()
    return fig, axes
