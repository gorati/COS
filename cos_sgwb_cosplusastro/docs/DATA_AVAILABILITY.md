# Data availability

The raw gravitational-wave strain data are not bundled in this repository.
The full pipeline fetches public H1 and L1 data through GWOSC-compatible tools
used by `gwpy` and `pygwb`.

The bundled archive contains only the final Bilby/Dynesty parameter-estimation
outputs and diagnostic plots for the completed run.

Not bundled:

- raw H1/L1 strain files;
- GWOSC cache files;
- intermediate `pygwb` point-estimate NPZ files;
- official LVK stochastic-background products.

A full rerun requires:

- internet access to GWOSC public data;
- a working `pygwb_pipe` command in the active environment;
- `gwpy`, `pygwb`, `bilby`, and `dynesty`.
