import itertools

import numpy as np
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params


def build_cnmf_params(
    fr,
    decay_time,
    gSig,
    K=40,
    merge_thr=0.85,
    p=2,
    nb=2,
    rf=None,
    stride_cnmf=None,
    bas_nonneg=True,
    min_SNR=0.1,
    rval_thr=0.1,
    cnn_thr=0.1,
    cnn_lowest=0.1,
    **overrides,
):
    """Build CNMFParams for source extraction and deconvolution.

    fr: frame (volume) rate in Hz.
    decay_time: indicator decay time constant, in seconds.
    gSig: expected half-width of a neuron in pixels, one value per spatial
        dimension (e.g. [14, 12.5, 5] for 3D). fr/decay_time/gSig are
        acquisition-specific and have no sensible default.
    **overrides: any additional CNMFParams keys (e.g. use_cnn, ssub, tsub),
        merged in as-is.
    """
    gSig = np.asarray(gSig)
    parameter_dict = {
        'fr': fr,
        'decay_time': decay_time,
        'p': p,
        'nb': nb,
        'gSig': gSig,
        'gSiz': gSig + 1,
        'rolling_sum': True,
        'only_init': True,
        'rf': rf,
        'stride': stride_cnmf,
        'merge_thr': merge_thr,
        'bas_nonneg': bas_nonneg,
        'min_SNR': min_SNR,
        'rval_thr': rval_thr,
        'use_cnn': False,
        'min_cnn_thr': cnn_thr,
        'cnn_lowest': cnn_lowest,
        'K': K,
        **overrides,
    }
    return params.CNMFParams(params_dict=parameter_dict)


def run_cnmf(images, opts, dview=None, n_processes=None, se=np.ones((3, 3, 1), dtype=np.uint8)):
    """Build a CNMF object from opts and fit it on images.

    se: structuring element used to dilate spatial footprints (default is
        shaped for 3D data as (y, x, z)); pass se=None to skip the override
        and use CNMFParams' own 'spatial' default.
    """
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    if se is not None:
        cnm.params.set('spatial', {'se': se})
    cnm.fit(images)
    return cnm


def grid_search_cnmf_params(images, param_grid, base_params, dview=None, n_processes=None):
    """Fit CNMF once per combination of param_grid values (Cartesian
    product), holding base_params fixed for everything else.

    param_grid: dict mapping build_cnmf_params keyword names to a list of
        values to try, e.g. {'K': [30, 40, 50], 'merge_thr': [0.7, 0.85]}.
    base_params: dict of the remaining build_cnmf_params arguments (must
        include fr, decay_time, gSig), merged into every combination.

    Returns a list of {'overrides': dict, 'n_components': int, 'cnm': CNMF},
    one per combination, for comparison in a notebook (e.g. via
    pandas.DataFrame). Each combination is a full CNMF fit, so this can take
    a long time for large grids -- it does not attempt to pick a "best"
    result; run evaluate_components yourself on whichever candidate looks
    promising.
    """
    keys = list(param_grid)
    combos = list(itertools.product(*param_grid.values()))
    results = []
    for i, values in enumerate(combos, 1):
        overrides = dict(zip(keys, values))
        print(f"[{i}/{len(combos)}] Fitting CNMF with {overrides}")
        opts = build_cnmf_params(**{**base_params, **overrides})
        cnm = run_cnmf(images, opts, dview=dview, n_processes=n_processes)
        results.append({
            'overrides': overrides,
            'n_components': cnm.estimates.A.shape[-1],
            'cnm': cnm,
        })
    return results
