# Gaussian Processes Joint Modeling simulation

An updated simulation notebook for studying shared latent dynamics across neural and behavioural observations with Gaussian Processes Joint Modeling (GPJM).

## What the notebook demonstrates

- generation of simulated neural and behavioural data;
- fitting a two-dimensional latent-state model;
- neural and behavioural predictions with uncertainty intervals; and
- comparison of recovered spatial, temporal and spatiotemporal kernels with simulation ground truth.

## Files

- `Simulation_GPJM_complete.ipynb` — updated runnable workflow;
- `GPJMv3.py` — model implementation required by the notebook;
- `GPJMv3_datagen.py` — simulation utilities; and
- `GPJMv3_functions.py` — supporting functions.

## Attribution

This work adapts and troubleshoots the GPJM implementation originally published by Giwon Bahg and collaborators at <https://github.com/giwonbahg/gpjm>. The upstream project should be cited when this adaptation is used. The files are provided here for research review and reproducibility; check the upstream project terms before redistribution.

## Reproducibility note

Notebook outputs and execution counts were removed from this public copy. The workflow depends on TensorFlow and GPflow APIs compatible with the original implementation and has not been executed in this portfolio build environment.
