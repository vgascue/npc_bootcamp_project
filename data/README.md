# Data folder structure

Raw imaging data is not tracked in Git (see `.gitignore`) but is expected to live
under this `data/` folder, or be pointed to via a `base_directory` argument that
follows the same layout.

## 3D (volumetric) time series data

For 3D acquisitions (a Z-stack captured at every timepoint), each imaging
session is organized as:

```
data/
└── YYYY_MM_DD_animal_x/
    └── raw_data/
        ├── TSeries-.../
        │   ├── ..._Ch1_0001.ome.tif
        │   ├── ..._Ch1_0002.ome.tif
        │   ├── ..._Ch2_0001.ome.tif
        │   ├── ..._Ch2_0002.ome.tif
        │   └── ...
        └── TSeries-.../
            └── ...
```

- `YYYY_MM_DD_animal_x/` — one folder per session, named with the acquisition
  date and animal ID (e.g. `2026_01_15_animal_3`). `create_output_directory`
  derives the output folder name from this level.
- `raw_data/` — contains one subfolder per acquisition run. `generate_file_list`
  looks for subfolders whose name contains `Tseries` (case-sensitive, matched
  against the folder name, not the full path).
- `TSeries-.../` — one folder per run. Each run is saved as a set of
  multipage OME-TIFF files, split by channel: `Ch1` is the reference
  (anatomical) channel and `Ch2` is the calcium (GCaMP) channel. Each file
  holds one Z-stack volume; the files are linked via OME-XML metadata into a
  single virtual series, so reading the first ("master") file for a channel
  with `tifffile.imread` transparently loads the full
  `(timepoints, z-planes, height, width)` volume for that run — this is why
  `generate_file_list` only needs to return the first `Ch2` file per folder.
