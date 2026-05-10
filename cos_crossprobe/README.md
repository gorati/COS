# COS Cross-Probe

This repository contains a falsification-oriented Pantheon+ → CMB cross-probe pipeline for testing possible COS time-arrow signatures.

**Current scientific status:** the reproduced run is compatible with the null hypothesis. The results should not be interpreted as a detection claim for a COS time-arrow signature.

## Repository scope

This repository is intended to provide a clean, reproducible public package for the current cross-probe workflow. It contains the active source code, the recommended shell wrappers, the required small input-data layer when redistribution is allowed, and the reproduced output products.

The intended public layout is:

```text
cos-crossprobe/
  README.md
  CITATION.cff
  requirements.txt
  environment.yml

  data/
    pantheon/
      README.md
    cmb/
      README.md

  src/
    cos_crossprobe/

  scripts/
    00_check_environment.sh
    01_pantheon_axis_scan.sh
    02_split_sample_validation.sh
    03_axis_recurrence_map.sh
    04_cmb_followup_recurrence_A.sh
    05_cmb_followup_recurrence_B.sh

  results/
    rerun.zip
    RESULTS_SUMMARY.md
```

## What the pipeline does

The workflow has four stages:

1. **Pantheon+ axis scan** — searches for candidate anisotropy/time-arrow axes in the Pantheon+ supernova sample.
2. **Split-sample validation** — tests whether the candidate-axis structure is stable under repeated train/validation splits.
3. **Axis recurrence map** — identifies recurrent candidate-axis families from the split-validation output.
4. **Fixed-axis CMB mutual-information follow-up** — evaluates whether the recurrent axes show a compatible CMB cross-scale mutual-information trend.

This is a stress-test pipeline. Its purpose is to make the COS empirical layer more falsifiable and auditable, not to force a positive result.

## Quick start

Run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/00_check_environment.sh
```

Run the Pantheon-only stages:

```bash
bash scripts/01_pantheon_axis_scan.sh
bash scripts/02_split_sample_validation.sh
bash scripts/03_axis_recurrence_map.sh
```

For the CMB follow-up, place the required external Planck/SMICA files into `data/cmb/`:

```text
COM_CMB_IQU-smica_2048_R3.00_full.fits
COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits
```

Then run:

```bash
bash scripts/04_cmb_followup_recurrence_A.sh
bash scripts/05_cmb_followup_recurrence_B.sh
```

Outputs are written to `results/rerun/`.

## Active code path

The active rerun chain is:

```text
src/cos_crossprobe/cos_pantheon_axis_scan.py
src/cos_crossprobe/cos_split_sample_crossprobe_standalone_checkpointed.py
src/cos_crossprobe/cos_axis_recurrence_map.py
src/cos_crossprobe/cos_crossprobe_fixed_axis_patched.py
```

## Reproduced result summary

The reproduced run did not produce robust positive cross-probe evidence. The key statistics are:

| Stage | Reproduced value | Interpretation |
|---|---:|---|
| Pantheon fixed-axis null p-value | 0.0858283 | Weak/borderline exploratory feature; not a detection |
| Split validation: median validation percentile | 0.492008 | Null-compatible |
| Split validation: mean validation percentile | 0.513936 | Null-compatible |
| Split validation: median fixed-axis null p | 0.360697 | No robust positive signal |
| Split validation: mean fixed-axis null p | 0.455970 | No robust positive signal |
| CMB recurrence axis A: `p_mono_decreasing` | 0.691542 | Null-compatible |
| CMB recurrence axis A: `p_slope_decreasing` | 0.422886 | Null-compatible |
| CMB recurrence axis B: `p_mono_decreasing` | 0.985075 | Does not support the chosen monotonic MI statistic |
| CMB recurrence axis B: `p_slope_decreasing` | 0.985075 | Does not support the chosen slope statistic |

The appropriate interpretation is:

> The Pantheon+ → split-validation → recurrence-map → fixed-axis CMB mutual-information pipeline is reproducible and currently null-compatible. It should be treated as a falsification-oriented stress test, not as positive evidence for a COS time-arrow signature.

## Data and redistribution note

The external CMB FITS files are not included in this repository. They must be obtained separately and placed under `data/cmb/` using the expected filenames above.

If Pantheon+ files are included under `data/pantheon/`, verify that redistribution is allowed by the relevant data-release terms before making the repository public. If redistribution is not allowed, remove the data files, keep checksums and file-name expectations, and provide download instructions instead.

## Intended COS publication role

This repository is most suitable as a supplement to the empirical/audit layer of the COS project. It should not be presented as the central positive-evidence result. Its value is that it documents a transparent cross-probe test that remains compatible with the null hypothesis.
