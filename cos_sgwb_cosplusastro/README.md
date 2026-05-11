# COS-SGWB Cos+Astro

This repository contains an exploratory stochastic gravitational-wave background
(SGWB) parameter-estimation prototype for the COS program.

The pipeline combines a standard astrophysical power-law SGWB component with an
analytic COS-inspired component,

```text
Omega_GW^COS(f) = Omega0 (f/f0)^alpha [1 + A_disc cos(omega ln(f/f_star) + phi)]
Omega_GW^astro(f) = Omega_astro0 (f/f0)^alpha_astro
```

and fits the model to H1--L1 stochastic-background point estimates using
`pygwb`, `gwpy`, `bilby`, and `dynesty`.

## Scientific status

The bundled run is **COS-relevant**, but it is **not a detection claim**.

The preferred interpretation is:

> This is a falsification-oriented SGWB stress test / upper-limit exercise. In
> the bundled O3a-style H1--L1 run, the data do not require an additional COS
> component. The result is compatible with the absence of a detectable COS SGWB
> signal in this configuration and constrains the allowed COS amplitude scale.

The result should not be presented as "COS detected in SGWB". It is best used
as a COS-NUM/COS-EXP supplement documenting how a COS+astrophysical SGWB model
can be constrained with public gravitational-wave data.

## Repository layout

```text
src/cos_sgwb_cosplusastro/sgwb_cosplusastro.py   Active pipeline
scripts/01_run_full_pipeline.*                   Full rerun wrappers
scripts/02_summarize_results.py                  Compact result summary
results/current/                                 Bundled PE outputs
results/current/summary_metrics.csv              Posterior/evidence summary
results/current/model_comparison.csv             Evidence comparison table
docs/                                            Methods, results, data notes
```

## Bundled results

The bundled run attempted 240 one-hour windows and retained 86 successful
windows, giving approximately 86 hours of effective H1--L1 livetime in the
combined SGWB point estimate. The final parameter-estimation outputs are:

- `cos_plus_astro_lvk` — preferred COS+astro model with LVK-scale priors;
- `astro_only_lvk` — comparison model with only an astrophysical power law;
- `cos_plus_astro_wide` — exploratory COS+astro model with wider amplitude priors.

Key model comparison:

| Comparison | Delta logZ | Bayes factor | Interpretation |
|---|---:|---:|---|
| `astro_only_lvk - cos_plus_astro_lvk` | 1.289 | 3.63 | weak preference for the simpler astro-only model |
| `cos_plus_astro_lvk - cos_plus_astro_wide` | 4.325 | 75.54 | LVK-scale priors are much better conditioned than the wide exploratory prior |

Main posterior summaries:

| Run | Parameter | Median | 90% interval |
|---|---|---:|---:|
| `cos_plus_astro_lvk` | `Omega0` | 3.526e-13 | [1.171e-13, 1.959e-12] |
| `cos_plus_astro_lvk` | `Omega_astro0` | 2.973e-13 | [1.131e-13, 1.280e-12] |
| `astro_only_lvk` | `Omega_astro0` | 3.263e-13 | [1.110e-13, 1.362e-12] |

Because the amplitude priors are log-uniform and the posterior is close to the
lower-prior/upper-limit regime, these values should be interpreted as
constraints/stress-test outputs, not as a robust measured signal.

## Quick start

Create an environment with the required gravitational-wave packages. A conda or
micromamba environment is recommended:

```bash
micromamba create -y -n cos-sgwb -c conda-forge python=3.11 numpy astropy gwpy pygwb bilby dynesty matplotlib pandas
micromamba activate cos-sgwb
```

Then run the full pipeline:

```bash
bash scripts/01_run_full_pipeline.sh
```

The full rerun requires internet access to public GWOSC data and a working
`pygwb_pipe` installation. It can be slow and sensitive to public-data server
availability.

To summarize the bundled JSON outputs without rerunning the full pipeline:

```bash
python scripts/02_summarize_results.py
```

## Disclaimer

This is research code. The included results are exploratory and should be
interpreted only with the data cuts, priors, frequency band, and GWOSC/pygwb
configuration documented here.
