import os
from pathlib import Path
import tifffile as tifffile
import caiman as cm
from caiman.source_extraction.cnmf import cnmf, params
import caiman.source_extraction.cnmf as cnmf
import caiman.paths
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


def convert_files_for_caiman(full_file_list):
    # Read all data as a single sequence
    image_sequence = tifffile.imread(full_file_list)
    print(np.shape(image_sequence))

    # Concatenate imaging runs end to end
    nds, nt, nz, ny, nx = np.shape(image_sequence)
    im_sequence_cat = image_sequence.reshape(nds * nt, nz, ny, nx)
    print(np.shape(im_sequence_cat))
    image_sequence = im_sequence_cat
    # Switch format to TXYZ for Caiman
    ca_movie = cm.movie(image_sequence)
    ca_movie = np.transpose(ca_movie, (0, 2, 3, 1))
    print(np.shape(ca_movie))
    # Save as h5 data for easier handling
    ca_movie.save(os.path.join(os.environ["CAIMAN_DATA"], 'CaData.h5'))
