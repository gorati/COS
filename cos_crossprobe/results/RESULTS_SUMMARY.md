# Results Summary

This document summarizes the reproduced COS Cross-Probe run intended for the public GitHub repository.

## Headline conclusion

The reproduced run is compatible with the null hypothesis. It does not provide robust positive evidence for a COS time-arrow signature.

This is still scientifically useful: the repository provides a reproducible stress-test pipeline that can be used as part of the empirical/audit layer of the COS project.

## Key reproduced metrics

| Quantity | Value | Interpretation |
|---|---:|---|
| Pantheon sample size | 620 | Expected Pantheon+ working sample size |
| Random axes in Pantheon scan | 1000 | Exploratory scan resolution |
| Pantheon fixed-axis null p-value | 0.0858283 | Weak/borderline exploratory feature; not a detection |
| Split validation: median validation percentile | 0.492008 | Null-compatible |
| Split validation: mean validation percentile | 0.513936 | Null-compatible |
| Split validation: median fixed-axis null p | 0.360697 | No robust positive signal |
| Split validation: mean fixed-axis null p | 0.455970 | No robust positive signal |
| CMB recurrence axis A: `p_mono_decreasing` | 0.691542 | Null-compatible |
| CMB recurrence axis A: `p_slope_decreasing` | 0.422886 | Null-compatible |
| CMB recurrence axis B: `p_mono_decreasing` | 0.985075 | Does not support the chosen monotonic MI statistic |
| CMB recurrence axis B: `p_slope_decreasing` | 0.985075 | Does not support the chosen slope statistic |

## Pantheon+ stage

The full Pantheon+ axis scan produces an exploratory feature with a fixed-axis null p-value of approximately `0.0858`. This is not small enough, and not independent enough after axis selection, to support a detection claim.

The result is useful as an input to the split-validation and recurrence-map stages, not as a standalone positive result.

## Split-sample validation

The split-validation stage is the strongest guard against over-interpreting an exploratory full-sample axis. The reproduced statistics are near the center of the null expectation:

```text
median validation percentile ≈ 0.492008
mean validation percentile   ≈ 0.513936
median fixed-axis null p     ≈ 0.360697
mean fixed-axis null p       ≈ 0.455970
```

These values indicate no stable, robust Pantheon+ axis signal in the current configuration.

## Recurrence map

The recurrence map identifies two axis families for follow-up:

```text
Recurrence axis A: approximately (l, b) = (111.31°, -16.64°)
Recurrence axis B: approximately (l, b) = (298.70°, 52.44°)
```

These are follow-up directions extracted from the split-validation structure. They should not be described as confirmed physical axes.

## CMB follow-up

The fixed-axis CMB mutual-information follow-up does not turn the Pantheon+ recurrence axes into a positive COS claim.

For recurrence axis A:

```text
p_mono_decreasing  ≈ 0.691542
p_slope_decreasing ≈ 0.422886
```

This is null-compatible.

For recurrence axis B:

```text
p_mono_decreasing  ≈ 0.985075
p_slope_decreasing ≈ 0.985075
```

This does not support the chosen monotonic or slope-based CMB mutual-information statistic.

## Interpretation

> The reproduced Pantheon+ → recurrence-map → fixed-axis CMB mutual-information cross-probe remains compatible with the null hypothesis. It is a useful falsification-oriented stress test, but it is not a detection claim for a COS time-arrow signature.

## What this result supports

The result supports the following limited claims:

- The current public workflow can be rerun.
- The reproduced outputs match the previously audited numerical interpretation.
- The pipeline is useful as a negative/neutral empirical stress test.
- The COS empirical layer benefits from including null-compatible tests rather than only positive-looking searches.

## What this result does not support

The result does not support the following claims:

- That a COS time-arrow signature has been empirically detected.
- That the recurrence axes are confirmed physical axes.
- That the CMB mutual-information follow-up provides positive evidence for COS.
- That the full COS model is validated by this pipeline.
