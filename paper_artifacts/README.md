# COS Publications - Python Code Bundle

This repository contains the Python source code featured in the COS publications. The source files are organized by module under `paper_artifacts/<module>/...`. The table below records the mapping between the publication source, the original LaTeX listing label, and the current repository path.

## Overview

This bundle includes Python scripts extracted from the COS publications and maintained as separate publication artifacts. The purpose of this index is to make the provenance of each extracted listing explicit and reviewer-traceable.

## Repository Structure

The code is structured as follows:

```text
paper_artifacts/
  cos_c/
  cos_qd/
  cos_qf/
  cos_susy/
```

## Publication ↔ artifact mapping

| Publication | Module | LaTeX label / reference | Original role in PDF | GitHub file | Repository path | Snapshot commit |
|---|---|---|---|---|---|---|
| `2_COS-QF_HU_v22_fixed.tex` | `COS-QF` | `lst:qf-curvature-table-bca` | Full Python listing extracted from the main text | `qf_curvature_table_bca.py` | `paper_artifacts/cos_qf/qf_curvature_table_bca.py` | `9e24582` |
| `2_COS-QF_HU_v22_fixed.tex` | `COS-QF` | `lst:qf-convergence-table-bca` | Full Python listing extracted from the main text | `qf_convergence_table_bca.py` | `paper_artifacts/cos_qf/qf_convergence_table_bca.py` | `9e24582` |
| `2_COS-QF_HU_v22_fixed.tex` | `COS-QF` | `lst:qf-vg-benchmark` | Full Python listing extracted from the main text | `qf_vg_benchmark.py` | `paper_artifacts/cos_qf/qf_vg_benchmark.py` | `9e24582` |
| `4_COS-QD_HU_v17_fixed.tex` | `COS-QD` | `lst:build-step-operator-ts2` | Full Python listing extracted from the main text | `build_step_operator_ts2.py` | `paper_artifacts/cos_qd/build_step_operator_ts2.py` | `9e24582` |
| `4_COS-QD_HU_v17_fixed.tex` | `COS-QD` | `lst:peripheral-spectrum` | Full Python listing extracted from the main text | `peripheral_spectrum.py` | `paper_artifacts/cos_qd/peripheral_spectrum.py` | `9e24582` |
| `4_COS-QD_HU_v17_fixed.tex` | `COS-QD` | `lst:path-sum-mh` | Full Python listing extracted from the main text | `path_sum_mh.py` | `paper_artifacts/cos_qd/path_sum_mh.py` | `9e24582` |
| `4_COS-QD_HU_v17_fixed.tex` | `COS-QD` | `lst:path-sum-mh-fit-demo` | Full Python listing extracted from the main text | `path_sum_mh_fit_demo.py` | `paper_artifacts/cos_qd/path_sum_mh_fit_demo.py` | `9e24582` |
| `6_COS-SUSY_HU_v15_fixed.tex` | `COS-SUSY` | `lst:susy-gw-aps-minipack` | Full Python listing extracted from the main text | `susy_gw_aps_minipack.py` | `paper_artifacts/cos_susy/susy_gw_aps_minipack.py` | `9e24582` |
| `8_COS-C_HU_v16_fixed.tex` | `COS-C` | `lst:cosc-pipeline` | Full Python listing extracted from the main text | `cosc_pipeline.py` | `paper_artifacts/cos_c/cosc_pipeline.py` | `9e24582` |

## Notes

- The `Snapshot commit` column points to the current `paper_artifacts` snapshot on `main` at the time this mapping was assembled.
- If the repository later receives a tagged release or DOI-backed archival snapshot, that release identifier should replace or complement the commit reference in publication-facing citations.
- Short didactic code snippets may still remain in the PDF when they serve an explanatory purpose; the table tracks only the extracted full-length Python artifacts.
