import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import h5py
import tifffile as tifffile
import glob as glob
import numpy as np

def generate_file_list(base_directory, search_string='TSeries', channel='Ch2'):
    """Return, for each matching T-series folder directly under base_directory,
    the sorted list of all per-cycle OME-TIFF files for the given channel.

    Each run's files must be passed to tifffile.imread() together (as a
    single list) to be stitched into the full (T, Z, Y, X) movie for that
    run -- OME metadata does not let a single "master" file stand in for
    the rest of the series.

    If `channel` is falsy, no channel filter is applied (use this for
    single-page acquisitions where channels are interleaved in one file
    rather than split across Ch1/Ch2 filenames).
    """
    run_file_lists = []
    pattern = f'*{channel}*' if channel else '*.tif*'
    # Find T-series folders in the base directory and grab all matching files for the given channel
    for entry in os.scandir(base_directory):
        if not entry.is_dir() or search_string not in entry.name:
            continue
        fnames = sorted(glob.glob(os.path.join(entry.path, pattern)))
        if not fnames:
            print(f"Warning: no {channel or 'matching'} files found in {entry.path}, skipping.")
            continue
        run_file_lists.append(fnames)
    return run_file_lists


def create_output_directory(base_directory, dest_directory):
    """Create (and return) an output directory named after base_directory's own folder name."""
    session_folder = Path(base_directory).name
    if not session_folder:
        raise ValueError("base_directory does not have a folder name to use for the output directory.")
    new_folder_path = Path(dest_directory) / session_folder
    new_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"New folder created at: {new_folder_path}")
    return new_folder_path


def save_h5(arr, path):
    """Write an array directly to HDF5, skipping the cm.movie overhead."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('mov', data=arr, chunks=True)


def _load_single_page_channels(run_file_lists):
    """Read, for each run, multipage OME-TIFFs with an interleaved channel
    axis; concatenate runs along time and split into (reference, calcium)
    movies, each in (T, Y, X, Z) order."""
    runs = []
    for run_files in run_file_lists:
        movie = tifffile.imread(run_files)
        if movie.ndim != 5:
            raise ValueError(
                f"Expected a 5D (T, channels, Z, Y, X) stack from "
                f"{len(run_files)} file(s), got shape {movie.shape}."
            )
        nc = movie.shape[1]
        if nc != 2:
            raise ValueError(f"Expected 2 interleaved channels (ref, ca), got {nc}.")
        runs.append(movie)

    combined = np.concatenate(runs, axis=0) if len(runs) > 1 else runs[0]
    ref = np.transpose(combined[:, 0], (0, 2, 3, 1))
    ca = np.transpose(combined[:, 1], (0, 2, 3, 1))
    return ref, ca


def _load_split_channel_files(ref_run_file_lists, ca_run_file_lists):
    """Read separate per-run reference/calcium OME-TIFF file sets in
    parallel, stitching each run's files into (T, Z, Y, X), concatenating
    runs along time, then transposing into (T, Y, X, Z) order."""
    def read_channel(run_file_lists):
        runs = []
        for run_files in run_file_lists:
            movie = tifffile.imread(run_files, maxworkers=4)
            if movie.ndim != 4:
                raise ValueError(
                    f"Expected a 4D (T, Z, Y, X) stack from {len(run_files)} "
                    f"file(s), got shape {movie.shape}."
                )
            runs.append(movie)
        combined = np.concatenate(runs, axis=0) if len(runs) > 1 else runs[0]
        return np.transpose(combined, (0, 2, 3, 1))

    print('Reading reference and calcium data in parallel...')
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_ref = ex.submit(read_channel, ref_run_file_lists)
        f_ca = ex.submit(read_channel, ca_run_file_lists)
        ref = f_ref.result()
        ca = f_ca.result()
    return ref, ca


def convert_files_for_caiman(
    base_directory,
    output_dir,
    single_page=False,
    search_string='TSeries',
    ref_channel='Ch1',
    ca_channel='Ch2',
    ref_filename='RefData.h5',
    ca_filename='CaData.h5',
):
    """Load reference/calcium channel movies for every matching T-series run,
    concatenate them along time, and save each channel as its own HDF5 file
    in CaImAn's expected (T, Y, X, Z) order (time first, Z last).

    Two on-disk layouts are supported:
    - single_page=True: each run is one set of multipage OME-TIFFs with both
      channels interleaved along a channel axis (no Ch1/Ch2 in the filename).
    - single_page=False (default): each run is split into separate Ch1
      (reference) and Ch2 (calcium) OME-TIFF file sets, as returned by
      `generate_file_list`.

    Returns the (ref_path, ca_path) the two channels were saved to.
    """
    output_dir = Path(output_dir)
    ref_path = output_dir / ref_filename
    ca_path = output_dir / ca_filename

    if single_page:
        full_file_list = generate_file_list(base_directory, search_string, channel=None)
        ref, ca = _load_single_page_channels(full_file_list)
    else:
        ref_file_list = generate_file_list(base_directory, search_string, channel=ref_channel)
        ca_file_list = generate_file_list(base_directory, search_string, channel=ca_channel)
        if len(ref_file_list) != len(ca_file_list):
            raise ValueError(
                f"Found {len(ref_file_list)} {ref_channel} run(s) but "
                f"{len(ca_file_list)} {ca_channel} run(s); expected one of each per T-series folder."
            )
        ref, ca = _load_split_channel_files(ref_file_list, ca_file_list)

    print('Writing reference and calcium data in parallel...')
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_ref = ex.submit(save_h5, ref, ref_path)
        f_ca = ex.submit(save_h5, ca, ca_path)
        f_ref.result()
        f_ca.result()

    return ref_path, ca_path
