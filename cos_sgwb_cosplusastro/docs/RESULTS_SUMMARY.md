# Results summary

## Scientific status

The bundled results are neutral to mildly adverse for an additional COS SGWB
component in this specific run. They are not a COS detection.

The main useful output is an upper-limit-style constraint on the COS amplitude
scale in the analyzed H1--L1 band.

## Evidence comparison

| Comparison | Delta logZ | Bayes factor | Interpretation |
|---|---:|---:|---|
| `astro_only_lvk - cos_plus_astro_lvk` | 1.289 | 3.63 | weak preference for astro-only |
| `cos_plus_astro_lvk - cos_plus_astro_wide` | 4.325 | 75.54 | LVK-scale prior is better conditioned |

The evidence does not support claiming a detected COS component.

## Posterior summaries

| Run | Parameter | Median | 90% interval |
|---|---|---:|---:|
| `cos_plus_astro_lvk` | `Omega0` | 3.526e-13 | [1.171e-13, 1.959e-12] |
| `cos_plus_astro_lvk` | `Omega_astro0` | 2.973e-13 | [1.131e-13, 1.280e-12] |
| `astro_only_lvk` | `Omega_astro0` | 3.263e-13 | [1.110e-13, 1.362e-12] |

The posterior amplitudes lie close to the low-amplitude prior/upper-limit
regime. They should not be read as precise astrophysical or COS detections.
