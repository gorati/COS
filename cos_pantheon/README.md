# COS-Pantheon Axis Scan

This repository/package contains a reproducible Pantheon+SH0ES Type Ia supernova
axis-scan prototype used in the COS empirical-audit layer.

The code tests whether low-redshift Pantheon+ residuals or fitted cosmographic
parameters show hemispherical asymmetry along a fixed COS axis or along scanned
random axes. The current preferred run uses a fixed low-redshift cosmographic
baseline and the `abs_delta_q0` hemisphere statistic, calibrated with a
sky-scramble null ensemble.

## Scientific status

This is **COS-relevant**, but it is **not** a positive detection claim. The
correct role is an exploratory / falsification-oriented Pantheon+ axis-scan
stage that can feed independent follow-up tests, such as the separate
`cos_crossprobe` Pantheon+--CMB workflow.

The current archived run gives:

- sample size: 620 Pantheon+SH0ES supernovae after the configured cuts;
- random axes: 1000;
- null realizations: 500 sky-scramble realizations;
- baseline fit: `q0 = -0.43`, `M = -19.345329`, `chi2/ndof = 550.879/618`;
- fixed COS-axis statistic: `abs_delta_q0 = 0.865`;
- fixed COS-axis percentile vs random axes: `70.6%`;
- fixed COS-axis null p-value: `0.17565`;
- scan-maximum global p-value: `0.02395`.

The scan maximum is exploratory and carries a look-elsewhere burden. It should
not be interpreted as a pre-registered discovery.

## Interpretation

- a Pantheon+ axis-scan module for COS-EXP empirical appendices;
- a precursor / input stage for the broader `cos_crossprobe` pipeline;
- a reproducibility supplement documenting how the Pantheon-side axis candidates
  were obtained.

A safer wording is:

> A Pantheon+ low-redshift axis scan was implemented as an exploratory
> falsification-oriented test. The fixed COS axis is null-compatible, while the
> scan maximum motivates independent cross-probe follow-up rather than a
> stand-alone detection claim.

## Repository layout

```text
cos_pantheon/
  README.md
  requirements.txt
  src/cos_pantheon/axis_scan.py
  scripts/00_download_pantheon_data.*
  scripts/01_run_axis_scan_q0_fixed.*
  data/pantheon/README.md
  results/current/pantheon_axis_scan_q0_fixed.json
  results/comparison/pantheon_axis_scan_metrics_summary.csv
```

## Quick start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the Pantheon+SH0ES inputs:

```bash
bash scripts/00_download_pantheon_data.sh
```

Run the current q0-fixed sky-scramble axis scan:

```bash
bash scripts/01_run_axis_scan_q0_fixed.sh
```

On Windows, use the corresponding `.bat` files.

## Data

The Pantheon+SH0ES input files are not redistributed by default in this cleaned
package. Download them from the official PantheonPlusSH0ES DataRelease
repository:

- `Pantheon+SH0ES.dat`
- `Pantheon+SH0ES_STAT+SYS.cov`
