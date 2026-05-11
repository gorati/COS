# COS-DM SPARC prototype (Collapsing-Structure Dark-sector / Dark-matter phenomenology)

This repository folder contains an exploratory COS-DM / SPARC rotation-curve prototype.
It is connected to the COS program as a **phenomenological effective-sector stress test**, not as a direct microscopic derivation of dark matter from COS geometry.

# COS-DM SPARC

**COS-DM** stands for **Collapsing Structure Dark-sector / Dark-matter
phenomenology**. The name denotes an exploratory COS-compatible effective
dark-sector program, tested here on SPARC galaxy rotation curves. It should not
be read as a claim that the COS framework already derives dark matter from its
microscopic structure.

## Scientific status

The current package should be read as an exploratory numerical supplement. Its main role is to ask what kind of baryon-coupled extra-acceleration law a later COS-compatible effective dark sector would need to reproduce on SPARC rotation curves.

The current preferred branch is the `g1` transition-scale modulation model:

```text
src/cos_dm_sparc/fit.py
scripts/01_run_g1.bat
scripts/01_run_g1.sh
results/current/g1/
```

The `hybrid` branch is kept as a control/comparison branch:

```text
scripts/02_run_hybrid.bat
scripts/02_run_hybrid.sh
results/current/hybrid/
```

Historical version labels are retained only under `archive/` and in provenance notes where they are needed to document the original development sequence.

## What this package is not

This package should **not** be presented as:

- a proof that COS derives dark matter;
- evidence that the literal COS skeleton is cold dark matter;
- a final physical dark-sector model;
- a detection claim.

The safer interpretation is:

> The SPARC prototype provides an effective phenomenological target: a COS-compatible emergent dark sector, if it exists, should reproduce a baryon-coupled extra acceleration with a global, galaxy-regime-dependent transition-scale modulation.

## Main files

```text
src/cos_dm_sparc/fit.py              Active fitter
scripts/00_dry_run_check.*          Parse/join check without fitting
scripts/01_run_g1.*                 Preferred current branch
scripts/02_run_hybrid.*             Control/comparison branch
data/sparc/                         SPARC input files used by the archived run
results/current/g1/                 Preferred g1 outputs
results/current/hybrid/             Hybrid comparison outputs
results/comparison/                 Historical comparison metrics table
docs/                               Method, results, data, provenance notes
archive/                            Legacy scripts/results/notes, retained only for provenance
```

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Check that the SPARC inputs can be parsed:

```bash
bash scripts/00_dry_run_check.sh
```

Run the preferred `g1` branch:

```bash
bash scripts/01_run_g1.sh
```

Run the comparison `hybrid` branch:

```bash
bash scripts/02_run_hybrid.sh
```

On Windows, use the corresponding `.bat` files:

```bat
scripts\00_dry_run_check.bat
scripts\01_run_g1.bat
scripts\02_run_hybrid.bat
```

## Current archived result summary

The archived `g1` run uses the full SPARC fitting set after the script's quality selection:

```text
n_galaxies = 171
n_points   = 3375
```

Main `g1` metrics:

```text
chi2_total                 ≈ 125570.02
median chi2/dof-like       ≈ 10.566
mean chi2/dof-like         ≈ 31.658
bad-fit fraction > 20      ≈ 0.3567
very-bad-fit fraction >100 ≈ 0.0702
```

The `hybrid` branch was run later by timestamp, but it is treated here as a comparison/control branch rather than the preferred current branch.

## Data

The included input files are from the public SPARC data release:

- `SPARC_Lelli2016c.mrt`
- `Rotmod_LTG.zip` and the extracted `Rotmod_LTG/*.dat` files

## Disclaimer

This is research code for exploratory model testing. It is not production-grade software, and the parametrization should not be treated as a final physical theory.
