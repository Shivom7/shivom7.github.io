# EEG preprocessing, P300 and microstate analysis

Research code for quality-controlled EEG preprocessing and an end-to-end P300 plus microstate workflow for OpenNeuro dataset `ds002893` (version 2.0.0).

## Included workflows

### Preprocessing notebook

`Preprocessed_Pipeline_Final_Marker_1.ipynb` covers EDF and marker loading, channel metadata and montage setup, bad-channel review, filtering, ICA-assisted ocular/cardiac artefact handling, average referencing, Welch-PSD quality control and cleaned FIF export.

### Full analysis pipeline

`Full_Pipeline_P300_Microstates.py` adds BIDS/EEGLAB discovery, automated channel QC and interpolation, memory-aware epoch preparation and caching, ERP/P300 feature extraction, GFP-peak clustering with `pycrostates`, cluster-count evaluation, template alignment, microstate backfitting, transition and complexity metrics, group statistics, mixed-effects analyses, multiple-comparison control, figures, manifests and reports. It supports command-line and graphical launch modes.

## Setup

Place the BIDS dataset at `data/ds002893-full`, or pass `--dataset`. Results default to `results`, or can be changed with `--output`. See `--help` for filters and run options.

## Reproducibility note

The notebook outputs and local machine paths were removed from this public copy. The Python files pass static syntax compilation, but scientific execution requires the dataset and the compatible MNE, pycrostates, NumPy, pandas, SciPy and statsmodels environment. Validate parameters and generated QC outputs before inferential use.
