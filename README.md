# JWST SED Anomaly Detection Pipeline

**UW Astronomy · Summer 2026 · Undergraduate Research Project**

A tsdat-inspired, four-stage pipeline for identifying galaxies with
statistically anomalous rest-frame spectral energy distributions (SEDs) in
archival JWST photometric catalogues — with a deliberately conservative
extension to galactic-scale biosignature and technosignature search limits.

---

## Table of Contents

1. [Abstract](#abstract)
2. [Research Questions](#research-questions)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Data Access](#data-access)
7. [Results Summary](#results-summary)
8. [Development & Testing](#development--testing)
9. [Citing This Pipeline](#citing-this-pipeline)
10. [Acknowledgements](#acknowledgements)
11. [Licence](#licence)

---

## Abstract

We present an unsupervised machine-learning pipeline for identifying galaxies
with statistically anomalous rest-frame spectral energy distributions (SEDs)
in archival JWST NIRCam/MIRI photometric catalogues (CEERS, JADES,
COSMOS-Web). Photometric redshifts and best-fit template SEDs are derived
with EAZY; per-band photometric residuals are then scored for anomalies
using an ensemble of Isolation Forest and UMAP+DBSCAN, and cross-matched
against known AGN and emission-line-galaxy catalogues to separate genuine
outliers from well-understood astrophysical contaminants. We characterise
the resulting anomaly rate as a function of redshift, stellar mass, and
star-formation rate, and interpret the residual "unexplained" population
within a deliberately conservative Bayesian framework for galactic-scale
technosignature and biosignature search limits. Even a well-characterised
null result — a demonstration that no statistically significant unexplained
anomaly population exists at current survey depths — constitutes an
independent methodological contribution and a quantitative upper limit on
non-standard astrophysical processes at cosmological scale.

*(Reproduced from `results/paper/main.tex`'s abstract, once drafted; this
copy is kept in sync by hand.)*

---

## Research Questions

1. Can an unsupervised ML pipeline reliably identify SED outliers in JWST catalogues?
2. What fraction of flagged outliers are explained by known astrophysical contaminants?
3. Does the anomaly rate show statistically significant redshift/mass/SFR dependence?

---

## Pipeline Architecture

Each stage is a direct analogue of a layer in the [tsdat](https://tsdat.readthedocs.io)
pipeline framework — noted in parentheses below — even though no tsdat
dependency is used directly.

```
                    MAST Catalogue (CEERS · JADES · COSMOS-Web)
                                    │
                                    ▼
   [Retriever]                                    pipeline/retriever.py
   (≈ tsdat retriever.yaml + reader)
                                    │  Raw DataFrame (ra, dec, z_phot, fluxes)
                                    ▼
   [Standardise / a1]                              pipeline/standardise.py
   (≈ tsdat dataset.yaml + converter)
                                    │  Residual Dataset (NetCDF, xarray, data level a1)
                                    ▼
   [Quality / b1]                                  pipeline/quality.py
   (≈ tsdat quality.yaml + checkers)
                                    │  Anomaly-Scored Dataset (NetCDF, data level b1)
                                    ▼
   [Output]                                        pipeline/output.py
   (≈ tsdat storage layer)
                                    │
                                    ▼
      FITS Catalogue  +  Diagnostic Figures  +  LaTeX Summary Table
      data/flagged/*.fits   results/figures/*.pdf   results/tables/*.tex
```

| Stage | Module | tsdat analogue | Output |
|---|---|---|---|
| Retriever | `pipeline/retriever.py` → `MASTRetriever` | `retriever.yaml` + reader | Raw `pandas.DataFrame` |
| Standardise | `pipeline/standardise.py` → `SEDStandardiser` | `dataset.yaml` + converter | Residual `xarray.Dataset`, data level **a1** |
| Quality | `pipeline/quality.py` → `apply_quality_pipeline` | `quality.yaml` + checkers | Anomaly-scored `xarray.Dataset`, data level **b1** |
| Output | `pipeline/output.py` → `CatalogueExporter`, `DiagnosticPlotter` | storage layer | FITS/CSV/NetCDF catalogue, figures, LaTeX table |

---

## Installation

### Option A — Dev container (recommended)

```bash
code ~/astronomy/astronomy.code-workspace
```

The dev container builds the `astro` conda environment automatically (see
`.devcontainer/environment.yml`) and forwards the ports needed for JupyterLab.

### Option B — Manual conda environment

```bash
conda env create -f ../.devcontainer/environment.yml
conda activate astro
pip install -r requirements.txt
```

### Option C — Plain pip (no conda; matches CI)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the major dependency versions (astropy, astroquery,
xarray, scikit-learn, umap-learn, eazy-py, pytest, …); exact conda-forge
pins live in `../.devcontainer/environment.yml`.

---

## Quickstart

### 1. Set your MAST API token (optional, for authenticated downloads)

```bash
export MAST_API_TOKEN=your_token_here
# Get a token at: https://auth.mast.stsci.edu/token
```

### 2. Run the full pipeline — one command

```bash
python -m pipeline.output
```

This chains all four stages (`retriever → standardise → quality → output`)
end-to-end via `pipeline.output.run_full_pipeline`, printing a tsdat-style
stage-transition log at each step and writing the final FITS/CSV/NetCDF
catalogue, diagnostic figures, and LaTeX table.

Or use the VS Code task: **⌘⇧B → 🔭 Run Full Pipeline**

### 3. Explore interactively

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
```

Start with `notebooks/03_anomaly_detection.ipynb` — the primary Phase 3
analysis document, fully runnable against a synthetic stand-in catalogue
until real MAST data has been retrieved.

---

## Data Access

| Survey | Archive | Catalogue | ~Size |
|--------|---------|-----------|-------|
| CEERS DR1 | [MAST](https://ceers.github.io/dr06.html) | `ceers_dr1.fits` | ~500 MB |
| JADES DR1 | [MAST HLSP](https://archive.stsci.edu/hlsp/jades) | `jades_dr1.fits` | ~1.2 GB |
| COSMOS-Web | [IRSA](https://cosmos.astro.caltech.edu/page/cosmosweb) | `cosmos_web_early.fits` | ~800 MB |

Data files are gitignored and never committed. Run `python -m pipeline.retriever`
to fetch and validate them directly, or let `run_full_pipeline` fetch them as
part of a full run. `MASTRetriever` tries `astroquery.mast` first and falls
back to a direct HTTPS download from the published data-release page if that
fails — see `pipeline/retriever.py` for the documented fallback behaviour.

---

## Results Summary

*Placeholder — to be filled in once Phase 3/4 analysis on real MAST catalogues completes.*

| Survey | N sources | Anomaly rate | AGN fraction | Unexplained | z-trend p-value |
|--------|-----------|-------------|--------------|-------------|----------------|
| CEERS | — | —% | —% | —% | — |
| JADES | — | —% | —% | —% | — |
| COSMOS-Web | — | —% | —% | —% | — |

---

## Development & Testing

```bash
pytest tests/ -v --cov=pipeline --cov-report=term-missing
```

Or use the VS Code task: **🧪 Run Tests**

All tests run against purely synthetic fixtures (`tests/conftest.py`) — no
network access or real catalogue downloads are required. CI
(`.github/workflows/ci.yml`) runs the full suite on every push and pull
request to `main` and enforces a minimum of 70% coverage on `pipeline/`.

---

## Citing This Pipeline

```bibtex
@software{aasha2026jwst,
  author  = {Aasha},
  title   = {{JWST SED Anomaly Detection Pipeline}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/<your-handle>/jwst-sed-anomaly},
  note    = {UW Astronomy, Summer 2026}
}
```

If citing the associated results, please cite the paper draft in
`results/paper/main.tex` once available.

---

## Acknowledgements

This research uses publicly available data from the MAST archive.
CEERS, JADES, and COSMOS-Web are supported by NASA JWST programme grants.
Pipeline architecture inspired by the [tsdat](https://tsdat.readthedocs.io) framework.

---

## Licence

MIT © 2026 Aasha
