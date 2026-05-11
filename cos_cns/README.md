# COS-CNS: Causality, Signal-Locality, and Finite-Speed Influence in Non-Unitary Discrete Spacetime Dynamics

This folder contains the reproducible numerical pipeline used for the COS-CNS paper:

**Causality, Signal-Locality (No-Signaling), and Finite-Speed Influence in Non-Unitary Discrete Spacetime Dynamics.**

The code implements a reference graph-based open-system dynamics used to test the operational CNS layer of the COS program. It is intended as a reproducible numerical/audit companion to the COS-CNS publication, not as a full COS-QD topological simulation.

---

## Scientific role

The COS-CNS pipeline tests whether a simple hard-local graph dynamics satisfies the expected operational causality checks:

- no-signaling outside the graph-local causal cone,
- finite-speed influence propagation on a discrete graph,
- scheduling/confluence diagnostics,
- non-confluent scheduling as a negative-control violation,
- NC1--NC4 operational controls,
- COS-NUM / COS-STAB-style run artifacts.

The reference dynamics is a qubit graph with matching-layer updates and trajectory-sampled amplitude damping. The hard-local cone is a baseline sanity check by construction; it should not be overinterpreted as a general Lieb--Robinson theorem for arbitrary COS-QD dynamics.

---

## Main files

### Active pipeline

- `cos_cns_pipeline_trajectories_graph.py`

  Main reproducible Python pipeline.

  It supports:

  - `chain`, `star`, and connected Erdos--Renyi (`er`) topologies,
  - no-signaling TVD tests,
  - hard-local cone heatmaps,
  - confluent vs non-confluent scheduling diagnostics,
  - NC1--NC4 controls,
  - COS-NUM-style run outputs,
  - COS-STAB-style `logs/timeseries.csv`,
  - publication figures copied to `figs/` for the selected `publish_topology`.

### Recommended publication runner

- `cos_cns_publication_autotune_then_run_chain.bat`

  Recommended Windows runner for the archived publication-style run.

  It performs two stages:

  1. a chain-only autotune/calibration run for the scheduling-control parameters;
  2. a fixed-parameter paper run on `chain`, `star`, and `er`.

  The included archived results were generated with this workflow.

### Convenience / direct runners

- `cos_cns_pipeline_trajectories_graph.bat`

  Direct Windows runner for the main pipeline.

  This is useful for a controlled rerun, but the exact archived publication result in `cos_cns_results.zip` corresponds to the two-stage `cos_cns_publication_autotune_then_run_chain.bat` workflow.

- `cos_cns_autotune_multi.bat`

  Multi-topology autotune / diagnostic runner.

- `cos_cns_autotune_multi_quick.bat`

  Reduced-size quick diagnostic runner.

### Archived results

- `cos_cns_results.zip`

  Archived run outputs used for publication/audit.

  The archive contains:

  - `runs/<run_id>/config.json`,
  - `runs/<run_id>/outputs/*.csv`,
  - `runs/<run_id>/outputs/*.npz`,
  - `runs/<run_id>/outputs/summary.json`,
  - `runs/<run_id>/logs/timeseries.csv`,
  - `runs/<run_id>/figs/*.pdf`,
  - top-level `figs/*.pdf` for the selected publication topology.

- `cos_cns_consol.log`

  Console log of the long publication-style run.

---

## Recommended quick start on Windows

From this folder, run:

```bat
cos_cns_publication_autotune_then_run_chain.bat
```

This first performs an autotune/calibration run on the `chain` topology, then runs the fixed paper configuration on:

```text
chain, star, er
```

The publication-style run is computationally expensive. On a desktop/workstation machine it may take many hours or longer, depending on CPU, memory, Python/NumPy configuration, and whether other workloads are active.

---

## Fixed paper-run command

The final fixed paper-run configuration selected by the autotune chain is:

```bat
python cos_cns_pipeline_trajectories_graph.py ^
  --topology_list chain,star,er ^
  --publish_topology chain ^
  --N 19 --steps 24 --trials 60 --ntraj 1200 ^
  --gamma 0.05 --seed 42 --edge_gate sqrt_swap ^
  --sched_steps 10 --sched_B 14 ^
  --eps_sched 1e-10 --eps_ns 1e-3
```

The `publish_topology` is set to `chain`, so the canonical publication figures copied to the top-level `figs/` directory correspond to the chain topology. The `star` and `er` runs are retained under `runs/<run_id>_*` as robustness topologies.

---

## Output layout

A typical run creates:

```text
runs/
  <run_prefix>/
    index.json

  <run_prefix>_chain/
    config.json
    logs/
      timeseries.csv
    outputs/
      nosignal.csv
      cone.npz
      sched_confluent.csv
      sched_nonconfluent.csv
      nc1.csv
      nc2.csv
      nc3.csv
      nc4.json
      summary.json
    figs/
      nosignal-tvd-vs-t.pdf
      cone-heatmap.pdf
      scheduling-variance.pdf
      fig-nc-nonlocal-raw.pdf
      fig-nc-nonlocal-filter.pdf
      fig-nc-scheduling.pdf
      fig-nc-postselection.pdf

  <run_prefix>_star/
    ...

  <run_prefix>_er/
    ...
```

The top-level `figs/` directory contains the publication-copied figure set for the selected `publish_topology`.

---

## Archived publication result

The archived result set in `cos_cns_results.zip` contains two relevant run families.

### 1. Autotune / calibration run

```text
20260224T200523Z_1ca76d_chain
```

This run was used to choose the scheduling-control parameters for the paper run. The selected values were:

```text
sched_steps = 10
sched_B     = 14
```

### 2. Fixed paper run

```text
20260422T004235Z_1ca76d_chain
20260422T004235Z_1ca76d_star
20260422T004235Z_1ca76d_er
```

The fixed paper run used:

```text
N             = 19
steps         = 24
trials        = 60
ntraj         = 1200
gamma         = 0.05
seed          = 42
edge_gate     = sqrt_swap
eps_ns        = 1e-3
eps_sched     = 1e-10
sched_steps   = 10
sched_B       = 14
topologies    = chain, star, er
publish       = chain
```

Summary of the archived paper run:

| Topology | dist(A,B) | Edges | Matchings | no-signaling | confluent scheduling | Overall |
|---|---:|---:|---:|---|---|---|
| `chain` | 18 | 18 | 3 | OK | OK | OK |
| `star` | 2 | 18 | 18 | OK | OK | OK |
| `er` | 2 | 73 | 11 | OK | OK | OK |

Key summary metrics from `outputs/summary.json`:

| Topology | `max_tvd_outside_cone` | `max_tvd_confluent` | `max_tvd_nonconfluent` | `NC4 tvd_uncond` | `NC4 tvd_cond` |
|---|---:|---:|---:|---:|---:|
| `chain` | 0.0 | 1.94e-16 | 0.03157 | 0.005 | 1.0 |
| `star` | 0.0 | 0.0 | 0.03225 | 0.005 | 1.0 |
| `er` | 0.0 | 1.94e-16 | 0.02774 | 0.005 | 1.0 |

Interpretation:

- the no-signaling outside-cone check passes in all three topologies;
- the confluent scheduling check is numerically stable;
- the non-confluent scheduling diagnostic shows the expected negative-control deviation;
- the NC4 conditional/post-selection control behaves as expected: unconditional TVD remains near zero, while conditional TVD is close to one.

---

## How to interpret these results

These runs support the COS-CNS operational layer in the following limited sense:

- the reference graph dynamics behaves as a hard-local, finite-speed model;
- no-signaling diagnostics pass outside the causal cone;
- schedule-confluent updates are stable;
- deliberately non-confluent / nonlocal controls produce violations as expected.

They should **not** be interpreted as:

- a full derivation of COS-QD dynamics,
- a proof of a general Lieb--Robinson bound,
- a complete quantum-gravity simulation,
- or an empirical detection claim.

The correct role of this code is as a reproducible numerical companion to the COS-CNS causality and signal-locality paper.

---

## Requirements

Minimal Python dependencies:

```text
numpy
matplotlib
```

Recommended environment:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib
```

Then run:

```bat
python cos_cns_pipeline_trajectories_graph.py --help
```

---

## Reproducibility notes

The script avoids Python's built-in `hash()` for seed generation, so `PYTHONHASHSEED` is not required for reproducibility.

Each run records:

- command parameters in `config.json`,
- per-topology summaries in `outputs/summary.json`,
- trajectory/scheduling/no-signaling outputs as CSV/NPZ/JSON,
- COS-STAB-style time-series logs in `logs/timeseries.csv`,
- publication figures as PDF files.

For archival use, cite the Git commit or Zenodo DOI corresponding to the exact version of this folder and the `cos_cns_results.zip` artifact.

---

## Suggested citation / publication placement

This code belongs primarily with:

- the COS-CNS paper as the direct reproducibility package;
- COS-NUM as a numerical/audit companion;
- COS-STAB as an example of run-artifact discipline and external auditability.

A suitable wording in the paper is:

> The COS-CNS numerical supplement provides a trajectory-based graph reference implementation for testing no-signaling, hard-local cone behavior, scheduling confluence, and NC1--NC4 operational controls. The archived paper run passes the no-signaling and confluent-scheduling checks for chain, star, and connected Erdos--Renyi topologies, while the non-confluent controls display the expected violations.

---

## Disclaimer

This is research code. It is intended for reproducible numerical experiments and publication auditing, not as production-grade simulation software.

Interfaces, defaults, filenames, and output layouts may change as the COS program evolves.
