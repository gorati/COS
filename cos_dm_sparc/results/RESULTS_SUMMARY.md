# Results summary

## Current preferred branch: g1

The current preferred branch is:

```text
results/current/g1/
```

Main metrics from `cos_dm_g1_report.json`:

```text
n_galaxies                  = 171
n_points                    = 3375
chi2_total                  ≈ 125570.02
median chi2/dof-like        ≈ 10.566
mean chi2/dof-like          ≈ 31.658
bad-fit fraction > 20       ≈ 0.3567
very-bad-fit fraction >100  ≈ 0.0702
```

## Control branch: hybrid

The `hybrid` branch is retained for comparison:

```text
results/current/hybrid/
```

It was generated later by timestamp, but it is not treated as the preferred current branch.

## Historical comparison

The historical comparison table is available at:

```text
results/comparison/cos_dm_sparc_metrics_summary.csv
```

Important interpretation:

- an earlier `g1` branch gives the lowest median chi2/dof-like among the archived table entries;
- another earlier `g1` branch gives the best raw chi2_total among the archived table entries;
- those earlier solutions are more affected by boundary/saturation behavior;
- the current `g1` branch is treated as the healthier parametrization, not as the best raw fit.

## Scientific interpretation

The robust qualitative message is:

1. a purely local slope/curvature correction is not sufficient;
2. the missing structure appears to be global and galaxy-regime dependent;
3. modulation of the transition scale `g1` is the cleaner current branch;
4. surface brightness remains a strong diagnostic channel;
5. the model is exploratory and should not be presented as a final COS-DM theory.
