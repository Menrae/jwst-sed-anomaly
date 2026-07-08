# Literature Index

Structured reference list for the JWST SED anomaly detection project, organized
by topic area. DOI fields are placeholders pending verification.

## 1. JWST Survey Programmes

| Authors | Year | Title | Journal | Relevance | DOI |
|---|---|---|---|---|---|
| Finkelstein, S. L. et al. | 2023 | CEERS: The Cosmic Evolution Early Release Science Survey | ApJL | Source of the CEERS NIRCam photometric catalog used for candidate selection. | `TODO-DOI` |
| Eisenstein, D. J. et al. | 2023 | Overview of the JWST Advanced Deep Extragalactic Survey (JADES) | arXiv preprint | JADES deep multi-band imaging/spectroscopy underlies the primary galaxy sample. | `TODO-DOI` |
| Casey, C. M. et al. | 2023 | COSMOS-Web: An Overview of the JWST Cosmic Origins Survey | ApJ | COSMOS-Web wide-area coverage provides the complementary large-volume sample. | `TODO-DOI` |
| Rieke, M. J. et al. | 2023 | Performance of NIRCam on JWST in Flight | PASP | Defines the instrument response and photometric calibration assumed in SED fits. | `TODO-DOI` |

## 2. SED Fitting Methods

| Authors | Year | Title | Journal | Relevance | DOI |
|---|---|---|---|---|---|
| Bruzual, G. & Charlot, S. | 2003 | Stellar Population Synthesis at the Resolution of 2003 | MNRAS | BC03 models form the stellar population synthesis basis for template fitting. | `TODO-DOI` |
| Boquien, M. et al. | 2019 | CIGALE: a python Code Investigating GALaxy Emission | A&A | CIGALE is the primary SED-fitting code used to derive physical parameters. | `TODO-DOI` |
| Brammer, G. B., van Dokkum, P. G. & Coppi, P. | 2008 | EAZY: A Fast, Public Photometric Redshift Code | ApJ | EAZY provides the photometric redshift estimates feeding the SED pipeline. | `TODO-DOI` |
| Conroy, C. | 2013 | Modeling the Panchromatic Spectral Energy Distributions of Galaxies | ARA&A | Review of SED-fitting methodology and systematic uncertainties informing pipeline design. | `TODO-DOI` |

## 3. Anomaly Detection in Astrophysical Data

| Authors | Year | Title | Journal | Relevance | DOI |
|---|---|---|---|---|---|
| Baron, D. & Poznanski, D. | 2017 | The Weirdest SDSS Galaxies: Results from an Outlier Detection Algorithm | MNRAS | Random Forest-based outlier detection on SDSS spectra motivates the anomaly detection approach. | `TODO-DOI` |
| Meusinger, H. et al. | 2012 | Unusual Emission-line and Absorption-line Quasars from SDSS | A&A | Precedent for identifying spectroscopically unusual quasar/AGN populations. | `TODO-DOI` |
| Reyes, E. et al. | 2023 | Machine Learning Outlier Detection in Photometric Galaxy Surveys | (journal TBD) | ML-based outlier detection framework closely parallels the project's anomaly pipeline design. | `TODO-DOI` |

## 4. Biosignature / Technosignature Frameworks

| Authors | Year | Title | Journal | Relevance | DOI |
|---|---|---|---|---|---|
| Wright, J. T., Mullan, B., Sigurdsson, S. & Povich, M. S. | 2014 | The Ĝ Infrared Search for Extraterrestrial Civilizations with Large Energy Supplies. I. Background and Justification | ApJ | Dysonian SETI framework provides the galactic-scale technosignature interpretive context for extreme anomalies. | `TODO-DOI` |
| Lingam, M. & Loeb, A. | 2019 | Colloquium: Physical constraints for the evolution of life on exoplanets | Reviews of Modern Physics | Establishes cosmological-scale astrobiology constraints relevant to interpreting anomalous signatures. | `TODO-DOI` |
| Lacki, B. C. | 2016 | A Priori Estimates on the Prior Probability of Detectable Biosignatures | arXiv preprint | Provides Bayesian prior reasoning for how (im)probable biosignature interpretations of anomalies should be weighted. | `TODO-DOI` |

---

## Key Concepts Glossary

**SED (Spectral Energy Distribution)**
The distribution of a source's electromagnetic flux as a function of wavelength or frequency, used to constrain physical properties (stellar mass, star formation rate, dust content, redshift) via template fitting.

**Photometric redshift**
An estimate of a source's redshift derived from broadband photometry (flux in a set of filters) rather than spectroscopy, by fitting the observed spectral shape against redshifted template SEDs.

**Rest-frame wavelength**
The wavelength of light as emitted in the source's own reference frame, obtained by correcting the observed wavelength for cosmological redshift (rest-frame = observed / (1 + z)).

**AGN contamination**
Flux contribution from an Active Galactic Nucleus (accreting supermassive black hole) that distorts a galaxy's SED away from a purely stellar-population signature, potentially biasing derived physical parameters or mimicking an anomaly.

**Dust attenuation law**
A parametrized relationship describing how interstellar/circumstellar dust scatters and absorbs light as a function of wavelength, applied to SED models to reproduce the reddening seen in observed galaxy spectra.

**Isolation Forest**
An unsupervised machine learning algorithm that detects anomalies by randomly partitioning feature space; anomalous points require fewer partitions to isolate and thus receive shorter average path lengths in the resulting tree ensemble.

**UMAP (Uniform Manifold Approximation and Projection)**
A nonlinear dimensionality reduction technique used to project high-dimensional data (e.g., SED features) into a lower-dimensional space for visualization or clustering while preserving local and global structure.

**Variational autoencoder (VAE)**
A generative neural network architecture that learns a probabilistic latent-space encoding of input data; reconstruction error or latent-space distance can be used to flag anomalous inputs that the model represents poorly.

**Biosignature**
An observable feature (e.g., a specific atmospheric gas or combination thereof) whose presence is interpreted as indicative of biological activity on a planetary body.

**Technosignature**
An observable feature indicative of technology produced by an extraterrestrial civilization (e.g., industrial pollutants, megastructures, artificial signals), distinct from naturally occurring biosignatures.

**Data level (a1/b1, tsdat convention)**
Designations from the `tsdat` data-processing framework indicating pipeline stage: `a1` denotes raw/ingested data after initial standardization, while `b1` denotes data after quality control and higher-level processing (e.g., derived quantities, corrections applied).
