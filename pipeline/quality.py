"""
quality.py — Stage 3: Anomaly Detection & QC
=============================================
Analogue of tsdat's quality.yaml + checker/handler architecture.

In tsdat, this stage runs a list of configured checkers against a
standardised Dataset and attaches `qc_*` variables recording the result of
each check, driven by a handler policy (flag vs. drop). Here the "checks"
are unsupervised anomaly detectors rather than sensor QC rules: instead of
"is this thermocouple reading physically plausible?", the question is "is
this galaxy's SED shape, relative to the rest of the survey, unusual?".

Applies unsupervised ML anomaly detection to SED residuals (from
`standardise.SEDStandardiser.extract_residuals`) and cross-matches against
known contaminant catalogues (AGN, emission-line aliasing) so that
downstream analysis can separate "anomalous but astrophysically explained"
from "anomalous and unexplained". Outputs the b1-level flagged dataset.

Config sourcing
----------------
`apply_quality_pipeline(ds, config_path)` takes `config_path` pointing at
`quality_config.yaml` (the tsdat-style checker/handler registry for this
stage — loaded and logged for auditability). The ML checker hyperparameters
(`isolation_forest`, `umap`, `dbscan`) and contaminant catalogue locations
live in the sibling `pipeline_config.yaml`'s `quality:` section (per Prompt
2.1/2.2's config schema — see `config/pipeline_config.yaml`), resolved
automatically as `config_path.parent / "pipeline_config.yaml"`.

Populated by: Claude Code Prompt 3.1 & 3.2
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from umap import UMAP

from pipeline.retriever import _to_snake_case
from pipeline.standardise import BAND_PIVOT_WAVELENGTH_UM, log_data_level_transition

logger = logging.getLogger(__name__)

# ── AGN catalogue column aliases (Milliquas / SDSS DR17 quasar catalogue) ───
_RA_ALIASES = ["ra", "raj2000", "ra_deg", "plug_ra"]
_DEC_ALIASES = ["dec", "dej2000", "dec_deg", "plug_dec"]


def _residual_matrix(ds: xr.Dataset) -> np.ndarray:
    """Extract the (sources x bands) residual matrix, NaN-safe.

    NaNs (bands with no valid photometry for a source) are filled with 0 —
    "no measured deviation" — rather than dropped, so every source
    contributes a fixed-width feature vector to the ML checkers below.
    """
    return np.nan_to_num(ds["residuals"].values, nan=0.0)


# ── 1. Abstract checker interface ────────────────────────────────────────


class AnomalyChecker(ABC):
    """Common interface for all anomaly detectors in this stage."""

    @abstractmethod
    def score(self, ds: xr.Dataset) -> np.ndarray:
        """Return one anomaly score per source (length == ds.sizes['source_id'])."""
        raise NotImplementedError


# ── 2. Isolation Forest ──────────────────────────────────────────────────


class IsolationForestChecker(AnomalyChecker):
    """Isolation Forest over the residual matrix.

    Returns scores in [0, 1] where 1 = most anomalous (i.e. sklearn's
    `score_samples`, which is *lower* for more abnormal points, is negated
    and min-max normalised).
    """

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    def score(self, ds: xr.Dataset) -> np.ndarray:
        X = _residual_matrix(ds)
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(X)
        raw = -model.score_samples(X)  # higher raw == more anomalous
        span = raw.max() - raw.min()
        if span == 0:
            return np.zeros_like(raw)
        return (raw - raw.min()) / span


# ── 3. UMAP + DBSCAN ──────────────────────────────────────────────────────


class UMAPDBSCANChecker(AnomalyChecker):
    """2D UMAP embedding of the residual matrix, outliers via DBSCAN noise.

    Returns a binary score (1 = DBSCAN noise point / outlier, 0 = clustered).
    As a side effect, stores the 2D embedding coordinates on `ds.attrs`
    (`umap_embedding_x`, `umap_embedding_y`) for downstream plotting.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        eps: float = 0.5,
        min_samples: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state

    def score(self, ds: xr.Dataset) -> np.ndarray:
        X = _residual_matrix(ds)
        n_samples = X.shape[0]

        # UMAP requires n_neighbors < n_samples; clip for small catalogues
        # (e.g. unit tests) rather than letting it raise.
        n_neighbors = max(2, min(self.n_neighbors, n_samples - 1))

        reducer = UMAP(
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        embedding = reducer.fit_transform(X)

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(embedding)

        ds.attrs["umap_embedding_x"] = embedding[:, 0].tolist()
        ds.attrs["umap_embedding_y"] = embedding[:, 1].tolist()

        return (labels == -1).astype(float)


# ── 4. Ensemble ───────────────────────────────────────────────────────────


class EnsembleChecker(AnomalyChecker):
    """Weighted average of one or more `AnomalyChecker` scores."""

    def __init__(self, checkers: List[AnomalyChecker], weights: List[float]) -> None:
        if len(checkers) != len(weights):
            raise ValueError(
                f"checkers and weights must be the same length "
                f"(got {len(checkers)} checkers, {len(weights)} weights)"
            )
        if not checkers:
            raise ValueError("EnsembleChecker requires at least one checker")
        self.checkers = checkers
        self.weights = weights

    def score(self, ds: xr.Dataset) -> np.ndarray:
        total_weight = sum(self.weights)
        weighted_sum = np.zeros(ds.sizes["source_id"])
        for checker, weight in zip(self.checkers, self.weights):
            weighted_sum = weighted_sum + weight * checker.score(ds)
        return weighted_sum / total_weight


class _PrecomputedScore(AnomalyChecker):
    """Wraps an already-computed score array as an `AnomalyChecker`.

    Lets `apply_quality_pipeline` reuse `EnsembleChecker`'s weighted-average
    logic for the final `qc_anomaly_score` without re-fitting the
    (potentially expensive) underlying models a second time.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=float)

    def score(self, ds: xr.Dataset) -> np.ndarray:
        return self._values


# ── 5. Contaminant cross-match ───────────────────────────────────────────


class ContaminantCrossMatch:
    """Flags sources whose anomalous SED is plausibly explained by a known
    astrophysical contaminant rather than a genuine anomaly."""

    def __init__(self) -> None:
        self.agn_catalogue: Optional[pd.DataFrame] = None

    def load_agn_catalogue(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load a known AGN catalogue (e.g. Milliquas or SDSS DR17 quasars).

        Accepts FITS or CSV; standardises column names to snake_case and
        canonicalises common ra/dec column aliases to `ra`/`dec`.
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix in (".fits", ".fit"):
            df = Table.read(filepath).to_pandas()
        elif suffix in (".csv", ".txt"):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported AGN catalogue format {suffix!r} for {filepath}")

        df = df.rename(columns={c: _to_snake_case(c) for c in df.columns})
        df = self._canonicalize_radec(df)

        if "ra" not in df.columns or "dec" not in df.columns:
            raise ValueError(
                f"AGN catalogue {filepath} has no recognisable ra/dec columns "
                f"(looked for {_RA_ALIASES} / {_DEC_ALIASES})"
            )

        self.agn_catalogue = df
        logger.info("Loaded AGN catalogue with %d entries from %s", len(df), filepath)
        return df

    @staticmethod
    def _canonicalize_radec(df: pd.DataFrame) -> pd.DataFrame:
        cols_lower = {c.lower(): c for c in df.columns}
        ra_col = next((cols_lower[a] for a in _RA_ALIASES if a in cols_lower), None)
        dec_col = next((cols_lower[a] for a in _DEC_ALIASES if a in cols_lower), None)
        rename = {}
        if ra_col and ra_col != "ra":
            rename[ra_col] = "ra"
        if dec_col and dec_col != "dec":
            rename[dec_col] = "dec"
        return df.rename(columns=rename) if rename else df

    def cross_match(self, df: pd.DataFrame, radius_arcsec: float = 1.5) -> pd.Series:
        """Sky cross-match `df` (needs `ra`/`dec` in degrees) against the
        loaded AGN catalogue. Returns a boolean Series aligned to `df.index`.
        """
        if self.agn_catalogue is None or self.agn_catalogue.empty:
            logger.warning("cross_match called with no AGN catalogue loaded; returning all-False")
            return pd.Series(False, index=df.index)

        source_coords = SkyCoord(ra=df["ra"].to_numpy() * u.deg, dec=df["dec"].to_numpy() * u.deg)
        agn_coords = SkyCoord(
            ra=self.agn_catalogue["ra"].to_numpy() * u.deg,
            dec=self.agn_catalogue["dec"].to_numpy() * u.deg,
        )
        _, sep2d, _ = source_coords.match_to_catalog_sky(agn_coords)
        matched = sep2d.arcsec <= radius_arcsec
        return pd.Series(matched, index=df.index)

    def flag_emission_line_galaxies(
        self,
        df: pd.DataFrame,
        line_rest_wavelength_um: float = 0.1216,
        band_fraction_tolerance: float = 0.05,
    ) -> pd.Series:
        """Flag sources where a strong emission line redshifts into a
        photometric band at the source's redshift, which can inflate that
        band's flux and masquerade as an SED anomaly.

        Defaults to Lyman-alpha (rest 0.1216 um / 1216 Angstrom). A line is
        considered "in-band" if its observed wavelength falls within
        `band_fraction_tolerance` of a band's pivot wavelength.
        """
        redshift_col = "z_phot" if "z_phot" in df.columns else "redshift"
        if redshift_col not in df.columns:
            raise ValueError("flag_emission_line_galaxies requires a 'z_phot' or 'redshift' column")

        observed_wavelength_um = line_rest_wavelength_um * (1.0 + df[redshift_col])

        flagged = pd.Series(False, index=df.index)
        for pivot in BAND_PIVOT_WAVELENGTH_UM.values():
            width = pivot * band_fraction_tolerance
            in_band = observed_wavelength_um.between(pivot - width, pivot + width)
            flagged = flagged | in_band
        return flagged


# ── 6. Full quality pipeline ─────────────────────────────────────────────


def apply_quality_pipeline(ds: xr.Dataset, config_path: Union[str, Path]) -> xr.Dataset:
    """Run the configured anomaly checkers + contaminant cross-match over
    an "a1" Dataset and return the flagged "b1" Dataset.

    Appends `qc_anomaly_score`, `qc_iso_forest_score`, `qc_umap_outlier`,
    `qc_agn_match`, `qc_emission_line_flag`, and promotes `data_level` from
    "a1" to "b1".
    """
    config_path = Path(config_path)
    with open(config_path) as fh:
        checker_registry = yaml.safe_load(fh) or {}
    logger.info(
        "apply_quality_pipeline: loaded %d checker definitions from %s",
        len(checker_registry.get("checkers", [])),
        config_path,
    )

    pipeline_config_path = config_path.parent / "pipeline_config.yaml"
    with open(pipeline_config_path) as fh:
        pipeline_config = yaml.safe_load(fh) or {}
    quality_cfg = pipeline_config.get("quality", {}) or {}

    anomaly_methods = quality_cfg.get("anomaly_methods", ["isolation_forest", "umap_dbscan"])
    iso_cfg = quality_cfg.get("isolation_forest", {}) or {}
    umap_cfg = quality_cfg.get("umap", {}) or {}
    dbscan_cfg = quality_cfg.get("dbscan", {}) or {}
    contaminant_cfg = quality_cfg.get("contaminant_catalogues", {}) or {}

    ds = ds.copy()
    n = ds.sizes["source_id"]

    precomputed_checkers: List[AnomalyChecker] = []
    weights: List[float] = []

    if "isolation_forest" in anomaly_methods:
        iso_checker = IsolationForestChecker(
            contamination=iso_cfg.get("contamination", 0.02),
            n_estimators=iso_cfg.get("n_estimators", 200),
            random_state=iso_cfg.get("random_state", 42),
        )
        iso_score = iso_checker.score(ds)
        precomputed_checkers.append(_PrecomputedScore(iso_score))
        weights.append(1.0)
    else:
        iso_score = np.full(n, np.nan)
    ds["qc_iso_forest_score"] = xr.DataArray(iso_score, dims=["source_id"])

    if "umap_dbscan" in anomaly_methods:
        umap_checker = UMAPDBSCANChecker(
            n_neighbors=umap_cfg.get("n_neighbors", 15),
            min_dist=umap_cfg.get("min_dist", 0.1),
            n_components=umap_cfg.get("n_components", 2),
            eps=dbscan_cfg.get("eps", 0.5),
            min_samples=dbscan_cfg.get("min_samples", 5),
            random_state=umap_cfg.get("random_state", 42),
        )
        umap_score = umap_checker.score(ds)
        precomputed_checkers.append(_PrecomputedScore(umap_score))
        weights.append(1.0)
    else:
        umap_score = np.full(n, np.nan)
    ds["qc_umap_outlier"] = xr.DataArray(umap_score, dims=["source_id"])

    if not precomputed_checkers:
        raise ValueError(
            "apply_quality_pipeline: quality.anomaly_methods in pipeline_config.yaml lists no "
            "recognised checkers (expected 'isolation_forest' and/or 'umap_dbscan')"
        )
    ensemble_score = EnsembleChecker(precomputed_checkers, weights).score(ds)
    ds["qc_anomaly_score"] = xr.DataArray(ensemble_score, dims=["source_id"])

    source_df = pd.DataFrame(
        {
            "ra": ds["ra"].values if "ra" in ds else np.full(n, np.nan),
            "dec": ds["dec"].values if "dec" in ds else np.full(n, np.nan),
            "z_phot": ds["z_phot"].values if "z_phot" in ds else np.full(n, np.nan),
        }
    )

    cross_match = ContaminantCrossMatch()
    agn_cfg = contaminant_cfg.get("agn", {}) or {}
    agn_catalogue_path = agn_cfg.get("catalogue_path")
    if agn_catalogue_path and Path(agn_catalogue_path).exists():
        cross_match.load_agn_catalogue(agn_catalogue_path)
        agn_match = cross_match.cross_match(
            source_df, radius_arcsec=agn_cfg.get("match_radius_arcsec", 1.5)
        )
    else:
        logger.warning(
            "apply_quality_pipeline: no local AGN catalogue found (quality.contaminant_catalogues"
            ".agn.catalogue_path in pipeline_config.yaml is unset or missing); qc_agn_match will "
            "be all-False. Download %s and set catalogue_path to enable cross-matching.",
            agn_cfg.get("name", "an AGN reference catalogue"),
        )
        agn_match = pd.Series(False, index=source_df.index)
    ds["qc_agn_match"] = xr.DataArray(agn_match.to_numpy(), dims=["source_id"])

    emission_flag = cross_match.flag_emission_line_galaxies(source_df)
    ds["qc_emission_line_flag"] = xr.DataArray(emission_flag.to_numpy(), dims=["source_id"])

    ds.attrs["data_level"] = "b1"
    ds.attrs["quality_pipeline_applied_at"] = datetime.utcnow().isoformat()

    log_data_level_transition("a1", "b1", n, n)
    return ds
