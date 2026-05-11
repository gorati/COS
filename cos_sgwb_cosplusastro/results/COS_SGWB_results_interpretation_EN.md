# COS-SGWB COS+Astro Run — Result Interpretation

Short summary

The present result is best described as neutral, or mildly constructive, rather
than negative for the COS program.

The run does not detect a statistically significant COS stochastic
gravitational-wave background (SGWB) component. However, it also does not
create a conflict with the COS framework. Its main scientific value is that it
turns the COS-SGWB idea into a concrete, reproducible inference problem and
places an upper-limit scale on the relevant COS amplitude parameters.

In short:

- no COS-SGWB detection is claimed;
- the data do not require an additional cosmological/COS SGWB component;
- the COS+astro model is not strongly excluded;
- the run provides a publishable upper-limit / stress-test result.

1. What does the run say about COS?

In the COS+astro run with the LVK-like prior, the cosmological COS-like SGWB
component is constrained to be small. The relevant amplitude scale is of order

    Omega_0 <~ 2 x 10^-12

at the quoted high-credibility upper-bound level used in the run summary.

The astrophysical SGWB amplitude is constrained to a comparable small scale in
this short-data configuration.

The model comparison mildly prefers the simpler astro-only model. The Bayes
factor in favor of the astro-only model is only of order 3--4, corresponding to
a weak preference, not to a decisive exclusion of the COS+astro model. The
correct interpretation is therefore:

    The current data do not require an additional COS-like cosmological SGWB
    component.

This is different from saying that the COS component is ruled out.

Operationally, the present result says that no COS-SGWB signal is visible in
this approximately 3.6-day effective H1--L1 data set. If such a component is
present in the tested frequency band, its contribution must remain at or below
the current upper-limit scale, roughly of order 10^-12 in Omega_0.

2. When would this be problematic for COS?

This result would become problematic only if the natural or default COS
parameter range predicted a much larger SGWB signal in the same frequency band,
for example

    Omega_0 ~ 10^-9 -- 10^-8,

and if that amplitude could not be reduced without creating tension with other
COS constraints, such as CMB, N_eff, or other cosmological bounds.

If independent COS calculations predict a typical SGWB amplitude well below
the current bound, for example in the 10^-13 -- 10^-15 range, then the present
H1--L1 result is fully compatible with the model. In that case the current
constraint is still relatively loose and leaves substantial parameter space
available.

3. Why is the result still useful?

For a new theoretical framework, the first empirical milestone is often not a
detection. It is the construction of a reproducible likelihood pipeline and the
derivation of a meaningful upper limit.

From this perspective, the run is useful and publishable:

- COS-NUM provides or motivates a concrete COS-SGWB spectral template.
- COS-EXP tests that template in an explicit SGWB inference pipeline.
- The result constrains the COS amplitude scale rather than claiming a
  detection.

A suitable interpretation is:

    Using an H1--L1 O3a-style SGWB analysis, we obtain an upper-limit scale of
    Omega_0 ~ 10^-12 for the tested COS-like SGWB component. The current data
    do not require the extra component, but the result is compatible with a
    broad range of low-amplitude COS parameter space.

This is valuable because it gives the COS program a concrete empirical
interface with gravitational-wave background data. It also demonstrates that
the COS-SGWB pipeline is operational, reproducible, and suitable for future
longer data sets.

The weak preference for the astro-only model should be reported honestly. It
means that the data do not justify the additional COS degree of freedom in the
current short-data run. It does not amount to a strong rejection of COS+astro.

4. Recommended wording for COS-NUM

In COS-NUM, the result should be presented mainly as a numerical and
reproducibility achievement:

    The COS+astro SGWB inference pipeline was implemented with nested sampling
    and run on a short H1--L1 data segment. The sampler, posterior summaries,
    checkpoint structure, and model-comparison outputs are stable enough to
    define a reproducible COS-SGWB analysis path.

Recommended emphasis:

- the nested-sampling workflow is functional for the COS+astro model;
- the posterior behaves sensibly under the chosen priors;
- some weakly constrained shape parameters remain prior-influenced;
- the method is ready to be applied to longer data sets and updated LVK
  releases.

Avoid claiming in COS-NUM that the SGWB data validate the COS model. The
appropriate claim is reproducibility and pipeline readiness.

5. Recommended wording for COS-EXP

In COS-EXP, the result should be presented as an empirical upper-limit and
stress-test result:

    No significant COS-SGWB signal is detected in the tested H1--L1 data set.
    The COS+astro model yields an upper-limit scale on the COS amplitude
    parameter Omega_0, while the simpler astro-only model is mildly preferred
    by the evidence.

Recommended emphasis:

- no significant detection;
- upper limits on Omega_0 and the astrophysical SGWB amplitude;
- weak astro-only preference, with Delta ln Z of order 1.3 against COS+astro
  in the reported comparison;
- no strong exclusion of the COS+astro model;
- the current constraint is compatible with low-amplitude COS-SGWB parameter
  ranges;
- longer O3, O4, and future LVK data are needed for a more sensitive test.

6. What not to claim

The following statements should be avoided:

- "COS-SGWB has been detected."
- "The data confirm the COS stochastic background."
- "The astro-only preference falsifies COS."
- "The current run proves or disproves the COS gravitational-wave sector."

The correct status is more modest and more defensible:

    This is the first reproducible COS-SGWB upper-limit / stress-test run in
    the COS+astro framework.

7. One-sentence summary

The current run is not a failure for COS: it does not detect a signal, but it
also does not contradict the model. It provides a first reproducible
upper-limit-scale constraint on the COS-SGWB amplitude and is suitable for
inclusion in COS-NUM and COS-EXP as a null-compatible empirical stress test.
