# template_project

## Overview

This project aims to provide a wrapper to use CaImAn for ROI segmentation of 2D and 3D data and manipulate the output ROIs. 

## Setup

```bash
mamba env create -f environment.yml
conda activate my_project
pip install -e .
```

## Project layout

```
├── src/my_project/     # importable package code
├── scripts/            # standalone scripts (e.g. run_analysis.py)
├── notebooks/          # exploratory notebooks
├── data/               # data files (see data/README.md)
└── tests/              # unit tests
```

## Usage

```bash
python scripts/run_analysis.py
```
