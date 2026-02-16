# COS-QSF CMB Pipeline

This folder contains the COS-QSF CMB analysis pipeline used to generate diagnostic spectra/plots
and numerical outputs for the COS-QSF module.

## What it does
- Runs a configurable CMB processing workflow (input maps / spectra / baseline CSV).
- Produces PDF figures and an `arrays.npz` bundle for downstream inspection.

## Requirements
- Python >= 3.10
- numpy, matplotlib, healpy (and its native dependencies)

Install:
```bash
pip install -r requirements.txt
