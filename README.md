# COS numerical pipelines and empirical tests

This repository collects research code related to the **COS** (**Collapsing
Structure**) framework. It focuses on numerical pipelines, empirical tests, and
reproducibility supplements used by the COS-NUM, COS-EXP, COS-CNS, and related
COS papers.

The repository contains both:

- reference or publication-facing numerical pipelines, and
- exploratory / falsification-oriented stress tests whose current results may be
  null-compatible rather than positive detections.

The main goals of this repository are:

- to provide **reference implementations** of key COS analysis pipelines;
- to document how the **CMB time-arrow**, **SGWB**, **Pantheon+**, **SPARC
  rotation-curve**, **LSS/DESI**, and **Pantheon+--CMB cross-probe** tests are
  computed in practice;
- to make it possible to **reproduce figures, tables, summaries, and
  diagnostic artifacts** in the COS-NUM, COS-EXP, COS-CNS, and related papers,
  given access to the corresponding public survey data;
- to preserve falsification-oriented, null-compatible, and upper-limit runs
  alongside positive or forecast-oriented analyses;
- to enforce **COS-STAB auditability** where applicable, using schema-checked
  `metrics.jsonl`, `run_meta.json`, `summary.json`, or equivalent run metadata;
- to keep a clear distinction between:
  - confirmed numerical behavior,
  - reproducible null results,
  - empirical upper limits,
  - exploratory phenomenology,
  - and speculative COS extensions.

The repository is designed to remain flexible. Over time, new scripts and
subprojects may be added as the COS program evolves.

---

## Scientific status of the included pipelines

Not every subproject has the same evidential status.

Some folders provide positive validation of a specific operational or numerical
layer, while others provide null-compatible stress tests or upper limits. In
particular:

- `cos_cns/` provides a positive numerical validation of the reference
  COS-CNS causality / no-signaling implementation.
- `cos_crossprobe/` provides a reproducible null-compatible Pantheon+--CMB
  cross-probe stress test.
- `cos_pantheon/` provides an exploratory Pantheon+ axis-scan / axis-candidate
  generator. It is not a stand-alone detection claim.
- `cos_sgwb_cosplusastro/` provides a COS+astrophysical SGWB inference pipeline
  and a current upper-limit / null-compatible result.
- `cos_dm_sparc/` provides an exploratory COS-DM effective dark-sector
  phenomenology prototype tested on SPARC rotation curves. It is not a
  derivation of dark matter from the COS microstructure.

This status separation is intentional. The repository is meant to support a
disciplined COS audit trail, not to overstate exploratory results.

---

## COS-STAB auditability

The folder `cos_stab/` contains the **COS-STAB audit layer** used by COS-NUM to
make numerical runs externally auditable. It provides a schema-enforcing logger,
an offline validator, and a combined reference runner.

Typical audit artifacts include:

```text
metrics.jsonl
run_meta.json
summary.json
config.json
timeseries.csv
```

Quick start for the audit demo:

```bash
cd cos_stab
python cos_core_sim_combined.py --steps 1000 --seed 2025 --lambda-geom 1.0 --output metrics.jsonl --run-meta run_meta.json
python validate_metrics.py metrics.jsonl --schema extended --strict
```

These artifacts and commands correspond to the COS-NUM auditability layer.

---

## Repository contents

The exact layout may change as the project is cleaned up and refactored, but
the main subprojects are:

### `cos_stab/`

COS-STAB audit logger, validator, and combined reference runner.

Role:

- numerical audit layer;
- schema-checked run metadata;
- reproducibility support for COS-NUM.

---

### `cos_cns/`

Reproducible numerics for the COS-CNS paper:

**Causality, Signal-Locality (No-Signaling), and Finite-Speed Influence in
Non-Unitary Discrete Spacetime Dynamics.**

Role:

- operational causality checks;
- no-signaling diagnostics;
- hard-local cone sanity checks;
- scheduling / confluence diagnostics;
- NC1--NC4 controls;
- COS-NUM / COS-STAB-style run artifacts.

The archived publication run passes the no-signaling and confluent-scheduling
checks for `chain`, `star`, and connected Erdos--Renyi topologies, while the
non-confluent controls display the expected violations.

See:

```text
cos_cns/README.md
```

for exact run commands, run IDs, and the mapping from scripts to paper figures.

---

### `cos_crossprobe/`

Falsification-oriented Pantheon+--CMB cross-probe pipeline for testing possible
COS time-arrow signatures across independent observational probes.

Workflow:

```text
Pantheon+ axis scan
→ split-sample validation
→ recurrence-map construction
→ fixed-axis CMB mutual-information follow-up
```

Current status:

- reproduced run is compatible with the null hypothesis;
- no COS time-arrow detection claim is made;
- suitable as an empirical stress test / audit supplement for COS-EXP and
  COS-NUM.

See:

```text
cos_crossprobe/README.md
```

for data requirements, run commands, and result interpretation.

---

### `cos_pantheon/`

Pantheon+ axis-scan pipeline.

Role:

- exploratory low-redshift Pantheon+ axis scan;
- fixed COS-axis test;
- sky-scramble null analysis;
- generation of axis candidates for follow-up tests such as `cos_crossprobe/`.

Current status:

- the fixed COS-axis result is null-compatible;
- the scan maximum is exploratory and look-elsewhere limited;
- this should not be interpreted as a stand-alone detection.

See:

```text
cos_pantheon/README.md
```

for the active q0-fixed run and Pantheon+SH0ES data instructions.

---

### `cos_sgwb_cosplusastro/`

COS+astrophysical stochastic gravitational-wave background (SGWB) inference
pipeline.

Role:

- combines an astrophysical SGWB component with a COS-like cosmological
  component;
- uses standard GW inference tools such as `pygwb`, `gwpy`, `bilby`, and
  `dynesty`;
- computes posterior summaries, evidences, Bayes factors, and upper-limit
  scales.

Current status:

- no significant COS-SGWB signal is detected in the current short H1--L1 run;
- the astro-only model is mildly preferred;
- the result is best interpreted as a null-compatible upper-limit / stress-test
  result, not as a COS-SGWB detection.

Important dependency note:

- this subproject is sensitive to the installed `pygwb`--`gwpy` version
  combination;
- use the subproject-specific `requirements.txt` / `environment.yml`;
- if using micromamba/conda, the recommended constraint is currently:

```text
gwpy<4
```

See:

```text
cos_sgwb_cosplusastro/README.md
```

for the exact environment and run instructions.

---

### `cos_dm_sparc/`

Exploratory COS-DM SPARC rotation-curve prototype.

Here **COS-DM** means:

```text
Collapsing Structure dark-sector / dark-matter phenomenology
```

Role:

- exploratory effective dark-sector phenomenology;
- SPARC galaxy rotation-curve fits;
- testing whether a COS-compatible emergent dark sector could reproduce a
  baryon-coupled extra-acceleration structure.

Current status:

- this is not a derivation of dark matter from COS microstructure;
- it should not be presented as "COS proves dark matter";
- it is a phenomenological target model and empirical stress test.

See:

```text
cos_dm_sparc/README.md
```

for the active model branch, data layout, and result interpretation.

---

### Legacy / standalone scripts

Some older or standalone scripts may remain in the repository root or in legacy
folders, including CMB time-arrow, COS-Planck, SGWB, or DESI-related scripts.
Examples may include:

- `cos_planck_v4_4_0.py`
- `cmb_time_arrow_MI_scan_axes.py`
- `cos_cmb_timearrow_bayes.py`
- `cos_desi_tests.py`

These scripts are retained for provenance or backward compatibility unless they
have been superseded by a cleaned subproject folder. Prefer the subproject
README files when a cleaned folder exists.

---

## Installation / environments

There is no single universal environment that is optimal for every subproject.
Some pipelines are lightweight, while others depend on gravitational-wave,
CMB, or nested-sampling packages with stricter version requirements.

Use an isolated environment for each major workflow.

### Lightweight pipelines

For simpler NumPy/Matplotlib-based pipelines:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install numpy scipy matplotlib pandas astropy
```

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip wheel
python -m pip install numpy scipy matplotlib pandas astropy
```

### CMB-oriented environment

For CMB / HEALPix-related workflows, install the required HEALPix backend:

```bash
micromamba create -y -n cmb -c conda-forge python=3.11 numpy scipy matplotlib pandas astropy tqdm healpy ducc0 dynesty bilby
micromamba activate cmb
```

### SGWB environment

For `cos_sgwb_cosplusastro/`, prefer the subproject-specific environment file.
A typical micromamba/conda setup is:

```bash
micromamba create -y -n cos-sgwb -c conda-forge python=3.11 numpy scipy matplotlib pandas astropy bilby dynesty "gwpy<4"
micromamba activate cos-sgwb
python -m pip install pygwb
```

Then run the subproject dependency check:

```bash
cd cos_sgwb_cosplusastro
bash scripts/00_check_environment.sh
```

If the SGWB run fails inside `pygwb.omega_spectra` with a `FrequencySeries`
`_print_slots` error, use the pinned environment recommended by the
`cos_sgwb_cosplusastro/` documentation.

### Subproject-specific environments

Individual subdirectories may provide their own:

```text
requirements.txt
environment.yml
environment_resolved.txt
```

Prefer those files over the generic examples above.

---

## Data

The COS pipelines rely on **public cosmological and astrophysical data sets**.
Large raw survey products are generally not bundled with this repository for
reasons of size, licensing, and provenance. Users should download maps, masks,
catalogs, covariance matrices, and strain data directly from the official
archives unless a subproject explicitly documents bundled data.

### Planck 2018 CMB maps and masks

For CMB-related COS-Planck and time-arrow analyses, the scripts may expect
Planck 2018 Release 3 component-separation maps and the common CMB mask.

Typical inputs include:

- SMICA IQU CMB map
- NILC IQU CMB map
- SEVEM IQU CMB map
- Commander IQU CMB map
- common CMB mask, INT, Nside=2048

The exact files, Nside values, smoothing conventions, masks, and preprocessing
steps are documented in the relevant subproject or paper.

### Pantheon+SH0ES data

The `cos_pantheon/` and `cos_crossprobe/` pipelines expect:

```text
Pantheon+SH0ES.dat
Pantheon+SH0ES_STAT+SYS.cov
```

These files should be downloaded from the official PantheonPlusSH0ES
DataRelease repository and placed under the path documented in the corresponding
subproject:

```text
cos_pantheon/docs/DATA_AVAILABILITY.md
cos_crossprobe/docs/DATA_AVAILABILITY.md
```

### SPARC data

The `cos_dm_sparc/` prototype uses SPARC galaxy rotation-curve data and mass
models.

The subproject documentation specifies whether the files are bundled or should
be downloaded from the official SPARC source. Before redistributing SPARC files
in a public fork, verify the applicable upstream data-use and citation
requirements.

### Gravitational-wave data

The `cos_sgwb_cosplusastro/` pipeline uses public LIGO/Virgo/KAGRA-style data
access through the standard GW software stack. The exact GPS windows, detector
pair, quality cuts, and `pygwb` configuration are documented in the subproject.

### DESI / LSS data

DESI and large-scale-structure scripts may require external catalogues, mocks,
or summary products. These are not stored in the repository unless explicitly
documented.

---

## Reproducibility and versioning

To support scientific reproducibility, the COS code follows these principles:

- each publication-facing subproject should include a local `README.md`;
- run scripts should explicitly specify input files, masks, seeds, priors, and
  numerical parameters;
- publication-facing outputs should include `config.json`, `summary.json`,
  `run_summary.json`, `model_comparison.csv`, `metrics.jsonl`, or equivalent
  metadata where appropriate;
- dependency-sensitive projects should include `requirements.txt`,
  `environment.yml`, or an equivalent environment record;
- COS-STAB auditability should be tracked by release tags, commit hashes, and
  immutable code snapshots when cited in papers;
- null-compatible and falsification-oriented runs should be preserved as part
  of the empirical audit trail, but explicitly documented as non-detection
  outcomes;
- upper-limit results should be described as upper limits, not as detections.

The COS-EXP, COS-NUM, COS-CNS, and related papers should cite this repository
or, preferably, an archived Zenodo DOI snapshot corresponding to the exact
version used.

---

## Interpretation of exploratory and null-compatible tests

Some pipelines in this repository are designed as exploratory probes or stress
tests rather than as direct evidence claims.

A null-compatible result should not be read as:

- a software failure;
- a positive detection;
- or a falsification of the entire COS framework.

Examples:

- `cos_crossprobe/` currently gives a null-compatible Pantheon+--CMB
  cross-probe result.
- `cos_pantheon/` gives an exploratory axis-scan result, not a detection.
- `cos_sgwb_cosplusastro/` gives a current SGWB upper-limit / stress-test
  result, not a COS-SGWB detection.
- `cos_dm_sparc/` gives effective dark-sector phenomenology, not a derivation
  of dark matter.

This distinction is intentional. The repository preserves positive,
null-compatible, and upper-limit results, provided that their status is clearly
documented.

---

## Recommended publication placement

The current subprojects map naturally to the COS paper series as follows:

| Subproject | Main publication role | Status |
|---|---|---|
| `cos_stab/` | COS-NUM / COS-STAB auditability | audit layer |
| `cos_cns/` | COS-CNS, COS-NUM supplement | positive operational validation |
| `cos_crossprobe/` | COS-EXP, COS-NUM supplement | null-compatible stress test |
| `cos_pantheon/` | COS-EXP exploratory axis scan | candidate generator, not detection |
| `cos_sgwb_cosplusastro/` | COS-EXP SGWB section, COS-NUM supplement | upper-limit / null-compatible |
| `cos_dm_sparc/` | COS-EXP or separate COS-DM supplement | exploratory phenomenology |

---

## Disclaimer

This repository is **research code**. While care has been taken to make the
pipelines transparent and reproducible, the scripts are not guaranteed to be
production-grade software. Interfaces, filenames, defaults, environments, and
directory structures may change as the COS program evolves.

Bug reports, questions, and suggestions are welcome via GitHub issues.
