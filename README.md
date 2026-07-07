# JWST SED Anomaly Detection Pipeline

**UW Astronomy · Summer 2026 · Undergraduate Research Project**

---

A data-driven pipeline for identifying galaxies with statistically anomalous
spectral energy distributions (SEDs) in archival JWST photometric catalogues
(CEERS, JADES, COSMOS-Web), with application to galactic-scale biosignature
and technosignature searches.

Even under the most conservative framing — a well-characterised null result —
the core pipeline is an independent contribution to the community and a basis
for a peer-reviewed publication.

---

## Pipeline Architecture

```
MAST Catalogue
    │
    ▼
[Retriever]          pipeline/retriever.py      (≈ tsdat retriever.yaml + reader)
    │  Raw DataFrame (ra, dec, z_phot, fluxes)
    ▼
[Standardise → a1]   pipeline/standardise.py   (≈ tsdat dataset.yaml + converter)
    │  Residual Dataset (NetCDF, xarray, data level a1)
    ▼
[Quality → b1]       pipeline/quality.py        (≈ tsdat quality.yaml + checkers)
    │  Anomaly-Scored + Flagged Dataset (NetCDF, data level b1)
    ▼
[Output]             pipeline/output.py
    │
    ├── data/flagged/*.fits         Science catalogue
    ├── results/figures/            Diagnostic plots
    └── results/tables/             LaTeX summary table
```

---

## Quickstart

### 1. Open in Dev Container

```bash
code ~/astronomy/astronomy.code-workspace
```

The dev container builds the `astro` conda environment automatically.

### 2. Set your MAST API Token

```bash
export MAST_API_TOKEN=your_token_here
# Get a token at: https://auth.mast.stsci.edu/token
```

### 3. Run the full pipeline

```bash
python -m pipeline.output
```

Or use the VS Code task: **⌘⇧B → 🔭 Run Full Pipeline**

### 4. Launch JupyterLab

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
```

---

## Data Access

| Survey | Archive | Catalogue | ~Size |
|--------|---------|-----------|-------|
| CEERS DR1 | [MAST](https://ceers.github.io/dr06.html) | `ceers_dr1.fits` | ~500 MB |
| JADES DR1 | [MAST HLSP](https://archive.stsci.edu/hlsp/jades) | `jades_dr1.fits` | ~1.2 GB |
| COSMOS-Web | [IRSA](https://cosmos.astro.caltech.edu/page/cosmosweb) | `cosmos_web_early.fits` | ~800 MB |

Data files are gitignored. Run `python -m pipeline.retriever` to fetch them.

---

## Installation (manual, without dev container)

```bash
conda env create -f ../.devcontainer/environment.yml
conda activate astro
pip install -r requirements.txt
```

---

## Research Questions

1. Can an unsupervised ML pipeline reliably identify SED outliers in JWST catalogues?
2. What fraction of flagged outliers are explained by known astrophysical contaminants?
3. Does the anomaly rate show statistically significant redshift/mass/SFR dependence?

---

## Results Summary *(placeholder — update after analysis)*

| Survey | N sources | Anomaly rate | AGN fraction | Unexplained | z-trend p-value |
|--------|-----------|-------------|--------------|-------------|----------------|
| CEERS | — | —% | —% | —% | — |
| JADES | — | —% | —% | —% | — |
| COSMOS-Web | — | —% | —% | —% | — |

---

## Citing This Pipeline

```bibtex
@software{aasha2026jwst,
  author  = {Aasha},
  title   = {JWST SED Anomaly Detection Pipeline},
  year    = {2026},
  url     = {https://github.com/<your-handle>/jwst-sed-anomaly},
  note    = {UW Astronomy, Summer 2026}
}
```

---

## Acknowledgements

This research uses publicly available data from the MAST archive.
CEERS, JADES, and COSMOS-Web are supported by NASA JWST programme grants.
Pipeline architecture inspired by the [tsdat](https://tsdat.readthedocs.io) framework.

---

## Licence

MIT © 2026 Aasha
