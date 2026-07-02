import caiman as cm
from pathlib import Path
from caiman.source_extraction.cnmf import params


def start_cluster(dview=None):
    """Stop any existing CaImAn cluster and start a fresh one.

    Returns (c, dview, n_processes); pass `dview` into functions that
    support parallel execution (e.g. run_motion_correction).
    """
    if dview is not None:
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)
    return c, dview, n_processes

def set_motion_correction_params(
    fnames,
    strides=(48, 48, 6),
    overlaps=(12, 12, 2),
    max_shifts=(4, 4, 2),
    max_deviation_rigid=5,
    pw_rigid=False,
    is3D=True,
):
    """Build CNMFParams for 3D motion correction.

    strides/overlaps/max_shifts are 3-tuples (x, y, z) in pixels; see the
    CaImAn NoRMCorre docs for tuning guidance.
    """
    opts_dict = {
        'fnames': fnames,
        'strides': strides,
        'overlaps': overlaps,
        'max_shifts': max_shifts,
        'max_deviation_rigid': max_deviation_rigid,
        'pw_rigid': pw_rigid,
        'is3D': is3D,
    }
    return params.CNMFParams(params_dict=opts_dict)

def run_motion_correction(opts, dview=None):
    """Run NoRMCorre motion correction using the fnames/params already stored in opts.

    Pass the `dview` returned by start_cluster() to parallelize across patches.
    """
    motion_params = dict(opts.get_group('motion'))
    fnames = motion_params.pop('fnames')
    mc = cm.motion_correction.MotionCorrect(fnames, dview=dview, **motion_params)
    mc.motion_correct(save_movie=True)
    return mc

def apply_shifts_to_ca(mc, ca_fname, output_dir):
    """Apply the reference-channel motion-correction shifts in `mc` to the
    calcium movie at `ca_fname`, saving both registered channels as HDF5.

    Returns (ref_moco_path, ca_moco_path).
    """
    output_dir = Path(output_dir)
    ref_moco_path = output_dir / 'RefData_Moco.h5'
    ca_moco_path = output_dir / 'CaData_Moco.h5'

    ref_moco = cm.load(mc.mmap_file, is3D=True)
    ref_moco.save(ref_moco_path)

    ca_moco = mc.apply_shifts_movie(ca_fname)
    ca_moco.save(ca_moco_path)

    return ref_moco_path, ca_moco_path

