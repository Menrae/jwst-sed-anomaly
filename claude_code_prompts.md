# Claude Code Prompt Set
## JWST SED Anomaly Detection Pipeline
### UW Astronomy — Summer 2026 Research Project

> **How to use this file:**
> Work through each phase sequentially in a Claude Code session. Each prompt is a
> self-contained instruction block — copy it verbatim into Claude Code. Prompts
> marked `[CHECKPOINT]` produce a testable artifact before moving on.
> The tsdat-style pipeline architecture (retriever → standardise → QC → output)
> is preserved throughout.

---

## PHASE 1 — Project Scaffold & Environment
### Weeks 1–2

---

### Prompt 1.1 — Repository Setup `[CHECKPOINT]`

```
Create a Python research project called `jwst-sed-anomaly` with the following
structure:

jwst-sed-anomaly/
  data/
    raw/          # downloaded MAST catalogues (gitignored)
    processed/    # SED-fit outputs and residuals
    flagged/      # anomaly-scored catalogues
  pipeline/
    __init__.py
    retriever.py      # Phase 2: data ingestion from MAST
    standardise.py    # Phase 2: SED fitting + residual extraction
    quality.py        # Phase 3: anomaly detection + contaminant QC
    output.py         # Phase 4: catalogue export + diagnostics
  notebooks/
    01_explore_catalogue.ipynb
    02_sed_fitting_demo.ipynb
    03_anomaly_detection.ipynb
    04_interpretation.ipynb
  tests/
    test_retriever.py
    test_standardise.py
    test_quality.py
  config/
    pipeline_config.yaml   # mirrors tsdat pipeline.yaml conventions
    quality_config.yaml    # mirrors tsdat quality.yaml conventions
  results/
    figures/
    tables/
  README.md
  requirements.txt
  .gitignore

For requirements.txt include: astropy, astroquery, numpy, pandas, scipy,
matplotlib, seaborn, scikit-learn, umap-learn, eazy-py, pyyaml, jupyter,
pytest, tqdm, h5py, requests.

For pipeline_config.yaml, create a tsdat-inspired schema with these top-level
keys: pipeline_name, data_level (set to "a1"), trigger_pattern, retriever,
standardiser, quality, storage. Populate each with sensible placeholder values
and inline comments explaining each field.

For quality_config.yaml, create a tsdat-inspired schema listing two placeholder
checkers: "sed_residual_range" and "photometric_coverage_fraction", each with
fields: variable, threshold, handler (set to "flag_and_log").

Write a README.md that explains the project purpose, the four pipeline stages,
and a quickstart section showing how to run the pipeline end-to-end.

Initialise the repo as a git repository and make an initial commit.
```

---

### Prompt 1.2 — Literature Reference Index

```
Inside the `results/` directory, create a file called `references.md`.

Populate it with a structured literature index for this project covering four
topic areas. For each paper include: authors, year, title, journal, a one-line
relevance note, and a placeholder DOI field.

Topic areas and papers to include:

1. JWST Survey Programmes
   - Finkelstein et al. 2023 (CEERS overview)
   - Eisenstein et al. 2023 (JADES overview)
   - Casey et al. 2023 (COSMOS-Web overview)
   - Rieke et al. 2023 (NIRCam instrument overview)

2. SED Fitting Methods
   - Bruzual & Charlot 2003 (BC03 stellar population models)
   - Boquien et al. 2019 (CIGALE)
   - Brammer et al. 2008 (EAZY)
   - Conroy 2013 (SED fitting review)

3. Anomaly Detection in Astrophysical Data
   - Baron & Poznanski 2017 (RF anomaly detection in SDSS spectra)
   - Meusinger et al. 2012 (unusual quasar spectra)
   - Reyes et al. 2023 (ML outlier detection in galaxy surveys)

4. Biosignature / Technosignature Frameworks
   - Wright et al. 2014 (Dysonian SETI, galactic-scale signatures)
   - Lingam & Loeb 2019 (astrobiology at cosmological scales)
   - Lacki 2016 (prior probability of detectable biosignatures)

After the reference list, add a section called "Key Concepts Glossary" defining:
SED, photometric redshift, rest-frame wavelength, AGN contamination, dust
attenuation law, Isolation Forest, UMAP, variational autoencoder, biosignature,
technosignature, data level (a1/b1 in the tsdat sense).
```

---

## PHASE 2 — Data Retrieval & SED Fitting
### Weeks 3–5

---

### Prompt 2.1 — MAST Catalogue Retriever `[CHECKPOINT]`

```
Implement `pipeline/retriever.py` as a production-quality Python module.

The module should contain a class `MASTRetriever` that mirrors the tsdat
Retriever concept: it is responsible solely for fetching and lightly
normalising raw input data, and must not perform any scientific analysis.

Requirements:

1. Method `fetch_ceers(output_path)`: uses astroquery.mast or direct HTTPS
   requests to download the CEERS DR1 photometric catalogue. If astroquery
   does not provide direct access, fall back to downloading from the published
   data release URL (https://ceers.github.io/dr06.html) and document the
   fallback clearly. Save the raw file to data/raw/ceers_dr1.fits or .csv.

2. Method `fetch_jades(output_path)`: same pattern for JADES DR1 from
   https://archive.stsci.edu/hlsp/jades.

3. Method `load_catalogue(filepath)` -> pd.DataFrame: reads either a FITS or
   CSV catalogue into a pandas DataFrame. Standardise column names to snake_case.
   Add a `survey` column with the source survey name. Return the DataFrame.

4. Method `validate_schema(df)` -> dict: checks that required columns are
   present (ra, dec, redshift or z_phot, and at least 4 photometric flux
   columns). Returns a dict with keys "valid" (bool), "missing_columns" (list),
   "n_sources" (int), "redshift_range" (tuple).

5. A module-level `REQUIRED_COLUMNS` list and `BAND_PREFIXES` list used for
   validation.

Add a `__main__` block that fetches both catalogues and prints the validation
report for each.

Write `tests/test_retriever.py` with at least three unit tests using pytest and
mock objects (do not make real network calls in tests).

This module is the direct analogue of tsdat's retriever.yaml + reader layer —
document this explicitly in the module docstring.
```

---

### Prompt 2.2 — SED Standardisation Layer `[CHECKPOINT]`

```
Implement `pipeline/standardise.py`.

This module is the analogue of tsdat's dataset.yaml + standardisation layer:
it takes raw input (the loaded catalogue) and produces a clean, science-ready
xarray Dataset with full metadata — the "a1" data level in tsdat terms.

Requirements:

1. Class `SEDStandardiser` with:

   a. `__init__(self, config_path)`: loads pipeline_config.yaml. Stores
      survey name, band list, redshift range filter, and min photometric
      coverage fraction as attributes.

   b. `preprocess(self, df)` -> pd.DataFrame: filters to sources with
      z_phot in [0.5, 10.0] (configurable), drops sources with fewer than
      4 valid flux measurements, converts fluxes to consistent units (uJy),
      and logs the number of sources removed at each step.

   c. `run_eazy_fit(self, df, output_dir)`: writes the EAZY input files
      (zphot.cat, zphot.param) from the DataFrame, runs EAZY as a subprocess,
      reads back the output .pz and .zout files, and returns a DataFrame of
      best-fit parameters (z_a, chi2, template_id) merged with the input.
      If EAZY is not installed, fall back to a stub that generates synthetic
      chi2 values from a lognormal distribution and logs a clear warning.

   d. `extract_residuals(self, df, fit_df)` -> xr.Dataset: computes per-band
      residuals (obs_flux - model_flux) / obs_flux_err for each source. Returns
      an xarray Dataset with dimensions (source_id, band). Attach metadata
      attributes: survey, data_level="a1", creation_timestamp, eazy_version,
      n_sources, n_bands. This mirrors how tsdat attaches global_attributes to
      its output Dataset.

   e. `save(self, ds, output_path)`: saves the xarray Dataset to NetCDF.
      Filename should follow tsdat's convention:
      `{survey}.a1.{YYYYMMDD}.{HHMMSS}.nc`

2. A module-level helper `log_data_level_transition(from_level, to_level, n_in,
   n_out)` that prints a standardised log line — analogous to tsdat's pipeline
   stage logging.

Write `tests/test_standardise.py` with tests for `preprocess` and
`extract_residuals` using synthetic DataFrames.
```

---

## PHASE 3 — Anomaly Detection & Quality Control
### Weeks 6–8

---

### Prompt 3.1 — Anomaly Detection Pipeline `[CHECKPOINT]`

```
Implement `pipeline/quality.py`.

This is the most scientifically critical module. It is the analogue of tsdat's
quality.yaml layer: it operates on the standardised Dataset and assigns quality
flags, but here the flags represent astrophysical anomaly scores rather than
sensor QC failures.

Requirements:

1. Abstract base class `AnomalyChecker` with method `score(ds: xr.Dataset) ->
   np.ndarray`. All detectors inherit from this.

2. Class `IsolationForestChecker(AnomalyChecker)`:
   - Fits sklearn IsolationForest on the residual matrix (sources x bands).
   - Returns anomaly scores in [0, 1] (1 = most anomalous).
   - Exposes `contamination` and `n_estimators` as constructor params with
     defaults matching the config file.

3. Class `UMAPDBSCANChecker(AnomalyChecker)`:
   - Reduces residual matrix to 2D with UMAP (n_neighbors=15, min_dist=0.1).
   - Runs DBSCAN on the embedding. Sources in cluster -1 (noise) are flagged.
   - Returns a binary score (0 = clustered, 1 = outlier).
   - Saves the 2D UMAP embedding coordinates as attributes on the Dataset.

4. Class `EnsembleChecker`:
   - Takes a list of AnomalyChecker instances and a list of weights.
   - Returns a weighted average anomaly score across all checkers.

5. Class `ContaminantCrossMatch`:
   - Method `load_agn_catalogue(filepath)`: loads a known AGN catalogue
     (e.g., from the Milliquas or SDSS DR17 quasar catalogue).
   - Method `cross_match(df, radius_arcsec=1.5)` -> pd.Series of booleans:
     performs a sky coordinate cross-match using astropy SkyCoord.
   - Method `flag_emission_line_galaxies(df)` -> pd.Series: flags sources
     where a strong emission line (e.g., Lyman-alpha) could alias into a
     photometric band at the source redshift and mimic an SED anomaly.

6. Function `apply_quality_pipeline(ds, config_path)` -> xr.Dataset:
   - Loads quality_config.yaml.
   - Instantiates the checkers listed in the config.
   - Appends anomaly scores as new variables: `qc_anomaly_score`,
     `qc_iso_forest_score`, `qc_umap_outlier`, `qc_agn_match`,
     `qc_emission_line_flag`.
   - Promotes data_level attribute from "a1" to "b1" on the Dataset.
   - This mirrors how tsdat's QC layer appends `qc_*` variables to the Dataset.

Write `tests/test_quality.py` with tests for IsolationForestChecker and
ContaminantCrossMatch using synthetic data.
```

---

### Prompt 3.2 — Exploration Notebook

```
Populate `notebooks/03_anomaly_detection.ipynb` as a fully runnable Jupyter
notebook. The notebook should serve as the primary analysis document for this
research phase.

Structure it with the following cells and narrative:

1. **Setup & Data Loading** — import the pipeline modules, load the b1 NetCDF
   Dataset, print a summary (n_sources, n_bands, redshift distribution).

2. **Residual Matrix Visualisation** — plot a heatmap of the residual matrix
   (sources x bands) using seaborn, sorted by redshift. Add a second plot
   showing the per-band residual distribution (violin plot).

3. **Isolation Forest Results** — run IsolationForestChecker, plot the
   distribution of anomaly scores, mark the top 5% threshold, and print how
   many sources are flagged.

4. **UMAP Embedding** — run UMAPDBSCANChecker, produce a scatter plot of the
   2D embedding coloured by anomaly score and a second scatter plot coloured
   by redshift. Discuss in a markdown cell what the cluster structure implies.

5. **Ensemble Score & Top Outliers** — compute ensemble scores, display a
   ranked table of the top 20 outliers with columns: source_id, ra, dec,
   z_phot, ensemble_score, qc_agn_match, qc_emission_line_flag.

6. **Contaminant Breakdown** — a stacked bar chart showing, for the top 5%
   of anomalies, what fraction are explained by: AGN, emission-line alias,
   poor photometric coverage, and "unexplained residual."

7. **Null Hypothesis Test** — a markdown cell describing the bootstrap
   permutation test to determine whether the anomaly rate is consistent with
   noise alone. Include a stub code cell that runs 1000 permutations on a
   synthetic dataset.

8. **Conclusions & Next Steps** — a markdown cell summarising findings and
   pointing to Phase 4.

Use synthetic/stub data where real catalogue data is not yet available, but
structure each cell so that swapping in real data requires only changing the
load path.
```

---

## PHASE 4 — Interpretation & Output
### Weeks 9–10

---

### Prompt 4.1 — Output & Catalogue Export `[CHECKPOINT]`

```
Implement `pipeline/output.py`.

This module is the tsdat storage layer analogue: it takes the fully processed
b1 Dataset and produces all final science outputs.

Requirements:

1. Class `CatalogueExporter`:
   - `to_fits(ds, output_path)`: writes the anomaly catalogue to a FITS
     BinTableHDU with all qc_* columns included and a primary HDU header
     containing the full pipeline metadata (survey, data_level, n_sources,
     creation_timestamp, pipeline_version).
   - `to_csv(ds, output_path)`: writes a clean CSV for human inspection,
     with qc_* columns and the top-level source metadata.
   - `to_netcdf(ds, output_path)`: saves the complete xarray Dataset as
     NetCDF following the tsdat filename convention:
     `{survey}.b1.{YYYYMMDD}.{HHMMSS}.nc`

2. Class `DiagnosticPlotter`:
   - `redshift_anomaly_rate(ds, output_dir)`: bins sources into redshift
     shells of dz=0.5. For each bin, computes anomaly rate (fraction with
     ensemble_score > 0.8). Plots rate vs. redshift with Poisson error bars.
     Saves as `results/figures/anomaly_rate_vs_redshift.pdf`.
   - `sky_distribution(ds, output_dir)`: plots RA/Dec of all sources with
     top outliers highlighted in a contrasting colour. Saves as
     `results/figures/sky_distribution.pdf`.
   - `score_summary_table(ds, output_dir)`: produces a LaTeX-formatted
     summary table (top 20 unexplained outliers) suitable for direct inclusion
     in a journal paper. Saves as `results/tables/top_outliers.tex`.

3. Function `run_full_pipeline(config_path)`:
   - Orchestrates all four pipeline stages end-to-end:
     retriever → standardise → quality → output.
   - Prints a tsdat-style stage transition log at each step.
   - Returns a dict with keys: n_sources_ingested, n_sources_processed,
     n_flagged, top_outlier_ids, output_paths.
   - This is the single entry point for running the entire summer project
     from one command: `python -m pipeline.output`
```

---

### Prompt 4.2 — Statistical Interpretation Analysis

```
In `notebooks/04_interpretation.ipynb`, build a statistical interpretation
analysis notebook.

1. **Redshift Evolution Test** — load the b1 Dataset. Bin sources into
   redshift shells dz=0.5 from z=0.5 to z=10. For each bin compute the
   anomaly rate. Fit a simple linear model (scipy.stats.linregress) to
   test whether anomaly rate correlates with redshift. Plot the fit with
   95% confidence intervals. Write a markdown cell stating the null hypothesis,
   test statistic, and p-value, and what the result implies.

2. **Stellar Mass & SFR Dependence** — if mass and SFR columns are available
   in the catalogue (they are in CEERS), produce a 2D hexbin plot of
   (log M_star, log SFR) coloured by median anomaly score. Discuss whether
   anomalous SEDs preferentially occur in a particular region of the
   star-forming main sequence.

3. **Unexplained Outlier Profiles** — for the top 10 sources with
   qc_agn_match=False and qc_emission_line_flag=False, generate a postage-stamp
   SED plot: observed fluxes with error bars, best-fit EAZY model, and residuals
   in a lower panel. Arrange all 10 as a 2x5 figure grid. Save as
   `results/figures/unexplained_outliers_seds.pdf`.

4. **Biosignature Prior Discussion** — a markdown cell walking through a
   simple Bayesian argument: given the number of galaxies surveyed, the
   observed anomaly rate, and the fraction unexplained by standard astrophysics,
   what is the posterior probability of a galactic-scale non-standard process
   (under a deliberately conservative prior)? Use sympy or plain arithmetic —
   the point is to show scientific rigour in the framing, not to claim a
   detection.

5. **Summary Statistics Table** — a final cell producing a publication-ready
   summary table as both a pandas DataFrame printout and a LaTeX snippet:
   survey, n_sources, anomaly_rate_percent, agn_fraction, emission_line_fraction,
   unexplained_fraction, redshift_trend_p_value.
```

---

## PHASE 5 — Write-Up & Dissemination
### Weeks 11–12

---

### Prompt 5.1 — Paper Draft Scaffold

```
Create a LaTeX paper scaffold at `results/paper/main.tex`.

The paper should follow the AASTeX 6.3 journal format (American Astronomical
Society — used by The Astronomical Journal and RNAAS). Structure it as follows:

- \documentclass[twocolumn]{aastex631}
- Standard AAS preamble packages (natbib, graphicx, amsmath, hyperref)
- Title: "A Search for Anomalous Spectral Energy Distributions in High-Redshift
  Galaxies with JWST: A Data-Driven Approach"
- Author placeholder with UW affiliation block
- Abstract (~200 word placeholder summarising: motivation, dataset, method,
  result, conclusion)
- Sections:
  1. Introduction (~3 paragraphs of placeholder text covering: JWST context,
     ML anomaly detection in astrophysics, astrobiology motivation)
  2. Data (subsections: CEERS, JADES, Data Reduction)
  3. Methods (subsections: SED Fitting, Residual Extraction, Anomaly Detection,
     Contaminant Removal)
  4. Results (subsections: Anomaly Rate, Redshift Evolution, Unexplained Outliers)
  5. Discussion
  6. Conclusions
  7. \acknowledgments — include placeholder for UW astronomy department,
     MAST archive, and any JWST programme IDs
  8. \bibliography{references}

Also create `results/paper/references.bib` with BibTeX entries for all papers
listed in `results/references.md` (use placeholder DOIs where real ones are
unknown).

Create a `Makefile` in `results/paper/` with targets: `pdf` (runs pdflatex
twice + bibtex), `clean`, and `arxiv` (zips source for arXiv submission).
```

---

### Prompt 5.2 — Poster Layout

```
Create a conference poster as an HTML file at `results/poster/poster.html`.

The poster should be A0 landscape (1189mm x 841mm) styled for UW's Autumn
Undergraduate Research Symposium using UW colours (purple #4B2E83, gold #B7A57A).

Layout (CSS Grid, 4 columns x 3 rows):

Row 1 (full width): Header — title, author name, UW logo placeholder,
  department, date

Row 2, Col 1: Motivation & Background — 3 bullet points
Row 2, Col 2: Data & Methods — flowchart as an HTML/SVG diagram showing
  the four pipeline stages (Retriever → Standardise → QC/Anomaly → Output)
  with tsdat-inspired stage labels
Row 2, Col 3: Results — placeholder for anomaly rate vs redshift figure
  (grey box with caption)
Row 2, Col 4: Top Outlier SEDs — placeholder for the 2x5 SED figure grid

Row 3, Col 1-2: Discussion & Conclusions — key takeaways as bullet points
Row 3, Col 3: Null Hypothesis Result — one large p-value number styled
  prominently with a brief interpretation sentence
Row 3, Col 4: QR code placeholder + references (5 key citations)

Style all text to be readable from 1.5m distance (minimum 24px body text).
Use print media query so the poster prints correctly to A0 at 96dpi.
```

---

## APPENDIX — Utility Prompts

These can be used at any point during the project.

---

### Prompt A.1 — Pipeline Config Updater

```
Read `config/pipeline_config.yaml` and `config/quality_config.yaml`. 

Update them to reflect the actual parameters used in the analysis so far:
- Replace all placeholder values with the real values used in the code
- Add a `last_updated` timestamp field
- Add a `git_commit_hash` field (retrieve via subprocess git rev-parse HEAD)
- Add a `checkers_used` list under quality config reflecting the actual
  AnomalyChecker subclasses instantiated
- Add a `data_provenance` block under pipeline config with the MAST download
  URLs and access dates

Print a diff of the changes made.
```

---

### Prompt A.2 — Test Suite & CI

```
Write a GitHub Actions workflow at `.github/workflows/ci.yml` that:
1. Runs on push and pull_request to main
2. Sets up Python 3.11
3. Installs requirements.txt
4. Runs pytest tests/ with --cov=pipeline --cov-report=xml
5. Fails if coverage drops below 70%

Also add a `conftest.py` in `tests/` with:
- A fixture `synthetic_catalogue` returning a 500-row DataFrame with columns
  ra, dec, z_phot, and 8 flux/flux_err columns (F115W, F150W, F200W, F277W,
  F356W, F410M, F444W, F770W) with realistic lognormal values
- A fixture `synthetic_residuals` returning an xarray Dataset built from
  synthetic_catalogue with qc_* variables pre-populated

These fixtures allow the full test suite to run without any network access or
real data downloads.
```

---

### Prompt A.3 — README Final Polish

```
Rewrite README.md as a polished, publication-quality project README.

Include:
1. A one-paragraph project abstract (copy from the LaTeX paper abstract)
2. A pipeline architecture diagram as ASCII art showing the four stages and
   their tsdat analogues in parentheses:
   MAST Catalogue → [Retriever] → Raw DataFrame
                  → [Standardise / a1] → Residual Dataset (NetCDF)
                  → [Quality / b1] → Anomaly-Scored Dataset (NetCDF)
                  → [Output] → FITS Catalogue + Figures + LaTeX Table
3. Installation instructions (conda env create, pip install -r requirements.txt)
4. Quickstart: one-command pipeline run
5. Data access instructions (MAST links, estimated download sizes)
6. Results summary table (fill with placeholder values)
7. Citation block in BibTeX format for citing this pipeline
8. Acknowledgements section
9. Licence: MIT
```

---

*Generated for UW Astronomy Summer 2026 Research Project*
*Pipeline architecture inspired by the tsdat time-series data framework*
