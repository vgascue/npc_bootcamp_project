import os
from pathlib import Path
import tifffile as tifffile
import glob as glob
import numpy as np

def generate_file_list(base_directory, search_string='Tseries', channel='Ch2'):
    """Return the master OME-TIFF path (default: Ch2 calcium channel) for each matching T-series folder."""
    full_file_list = []
    # Find T-series folders in the base directory and grab the master OME-TIFF for the given channel
    for entry in os.scandir(base_directory):
        if not entry.is_dir() or search_string not in entry.name:
            continue
        fnames = sorted(glob.glob(os.path.join(entry.path, f'*{channel}*')))
        if not fnames:
            print(f"Warning: no {channel} files found in {entry.path}, skipping.")
            continue
        full_file_list.append(fnames[0])
    return full_file_list


def create_output_directory(base_directory, dest_directory):
    """Create (and return) an output directory named after base_directory's parent (session) folder."""
    session_folder = Path(base_directory).parent.name
    if not session_folder:
        raise ValueError("The source directory does not have enough subfolders.")
    new_folder_path = Path(dest_directory) / session_folder
    new_folder_path.mkdir(parents=True, exist_ok=True)
    print(f"New folder created at: {new_folder_path}")
    return new_folder_path


def convert_files_for_caiman(full_file_list, output_dir=None, filename='CaData.h5'):
    """Load per-run 3D OME-TIFF stacks, concatenate them along time, and save
    as HDF5 in CaImAn's expected (T, Y, X, Z) order (time first, Z last).
    """
    image_sequence = tifffile.imread(full_file_list)
    if image_sequence.ndim != 5:
        raise ValueError(
            f"Expected a 5D (runs, T, Z, Y, X) stack from {len(full_file_list)} "
            f"file(s), got shape {image_sequence.shape}."
        )
    n_runs, nt, nz, ny, nx = image_sequence.shape

    # Concatenate runs end to end along the time axis
    image_sequence = image_sequence.reshape(n_runs * nt, nz, ny, nx)

    # CaImAn expects (T, Y, X, Z): time first, Z (depth) last
    ca_movie = cm.movie(image_sequence)
    ca_movie = np.transpose(ca_movie, (0, 2, 3, 1))

    output_dir = output_dir or os.environ["CAIMAN_DATA"]
    save_path = os.path.join(output_dir, filename)
    ca_movie.save(save_path)
    return save_path
