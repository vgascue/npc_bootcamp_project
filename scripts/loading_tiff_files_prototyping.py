import os
import tifffile as tifffile
import caiman as cm
from caiman.source_extraction.cnmf import cnmf, params
import caiman.source_extraction.cnmf as cnmf
import caiman.paths
import glob as glob
import numpy as np


def generate_file_list(base_directory, search_string='Tseries'):
    # base_directory = '/projectnb/mylabscc/Jack/Data/IR25_x_Gcamp1/20241015_Ir25_Gcamp/RawData/'

    full_file_list = []
    folder_names = []
    # Find T-series folders in the base directory
    for item in os.listdir(base_directory):
        # Create the full path
        item_path = os.path.join(base_directory, item)
        # Check if the item is a directory (and not a subdirectory)
        if os.path.isdir(item_path):
            print(item_path)
            folder_names.append(item_path)
    # Get the name of the OME tiff master file (first Ch1 file)
    for folder in folder_names:
        if search_string in folder:
            # print(folder)
            fnames_ref = glob.glob(os.path.join(folder, '*Ch2*'))
            fnames_ref.sort()
            full_file_list.append(fnames_ref[0])

    return full_file_list


def create_output_directory(base_directory, dest_directory):

    # Split the source directory into parts
    parts = base_directory.split(os.sep)

    # Check if there are enough parts to get the second to last folder
    if len(parts) < 2:
        raise ValueError("The source directory does not have enough subfolders.")
    # Get the second to last folder name
    second_last_folder = parts[-3]
    # Create the new folder path in the destination directory
    new_folder_path = os.path.join(dest_directory, second_last_folder)
    # Create the new folder
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"New folder created at: {new_folder_path}")


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
