"""
test_quality.py — Unit tests for pipeline.quality
====================================================
Uses purely synthetic Datasets/DataFrames; no real catalogue downloads.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pipeline.quality import (
    AnomalyChecker,
    ContaminantCrossMatch,
    EnsembleChecker,
    IsolationForestChecker,
    apply_quality_pipeline,
)

BANDS = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W"]
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _synthetic_residual_dataset(n=60, seed=0, n_outliers=4) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    residuals = rng.standard_normal((n, len(BANDS)))

    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    residuals[outlier_idx] *= 12.0  # inject extreme SED residuals

    ds = xr.Dataset(
        {
            "residuals": (["source_id", "band"], residuals),
            "chi2_eazy": (["source_id"], rng.lognormal(0.5, 0.7, size=n)),
            "z_a": (["source_id"], rng.uniform(0.5, 8.0, size=n)),
            "template_id": (["source_id"], rng.integers(1, 13, size=n)),
            "z_phot": (["source_id"], rng.uniform(0.5, 8.0, size=n)),
            "ra": (["source_id"], rng.uniform(214.8, 215.0, size=n)),
            "dec": (["source_id"], rng.uniform(52.8, 53.0, size=n)),
        },
        coords={"source_id": np.arange(n), "band": BANDS},
        attrs={"survey": "ceers", "data_level": "a1", "n_sources": n, "n_bands": len(BANDS)},
    )
    return ds, outlier_idx


# ── IsolationForestChecker ────────────────────────────────────────────────


def test_isolation_forest_checker_scores_in_unit_range():
    ds, _ = _synthetic_residual_dataset(n=60, seed=1)
    checker = IsolationForestChecker(contamination=0.1, n_estimators=100, random_state=42)

    scores = checker.score(ds)

    assert scores.shape == (60,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_isolation_forest_checker_flags_injected_outliers_higher():
    ds, outlier_idx = _synthetic_residual_dataset(n=80, seed=2, n_outliers=5)
    checker = IsolationForestChecker(contamination=0.1, n_estimators=200, random_state=42)

    scores = checker.score(ds)
    mean_outlier_score = scores[outlier_idx].mean()
    mean_inlier_score = np.delete(scores, outlier_idx).mean()

    assert mean_outlier_score > mean_inlier_score


def test_isolation_forest_checker_stores_constructor_params():
    checker = IsolationForestChecker(contamination=0.05, n_estimators=150, random_state=7)
    assert checker.contamination == 0.05
    assert checker.n_estimators == 150
    assert checker.random_state == 7


def test_isolation_forest_checker_default_params_match_config():
    checker = IsolationForestChecker()
    assert checker.contamination == 0.02
    assert checker.n_estimators == 200


# ── EnsembleChecker ───────────────────────────────────────────────────────


class _ConstantChecker(AnomalyChecker):
    """Tiny stub returning a fixed score, for testing EnsembleChecker in isolation."""

    def __init__(self, value):
        self.value = value

    def score(self, ds):
        return np.full(ds.sizes["source_id"], self.value)


def test_ensemble_checker_weighted_average():
    ds, _ = _synthetic_residual_dataset(n=10, seed=3)
    ensemble = EnsembleChecker([_ConstantChecker(0.0), _ConstantChecker(1.0)], [1.0, 3.0])

    scores = ensemble.score(ds)

    assert np.allclose(scores, 0.75)  # (0*1 + 1*3) / 4


def test_ensemble_checker_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        EnsembleChecker([_ConstantChecker(0.0)], [1.0, 2.0])


# ── ContaminantCrossMatch ─────────────────────────────────────────────────


@pytest.fixture
def agn_catalogue_csv(tmp_path) -> Path:
    path = tmp_path / "agn_catalogue.csv"
    pd.DataFrame({"RA": [214.85, 214.95], "DEC": [52.85, 52.95]}).to_csv(path, index=False)
    return path


def test_load_agn_catalogue_standardises_radec_columns(agn_catalogue_csv):
    matcher = ContaminantCrossMatch()
    df = matcher.load_agn_catalogue(agn_catalogue_csv)

    assert {"ra", "dec"}.issubset(df.columns)
    assert len(df) == 2
    assert matcher.agn_catalogue is not None


def test_cross_match_flags_sources_within_radius(agn_catalogue_csv):
    matcher = ContaminantCrossMatch()
    matcher.load_agn_catalogue(agn_catalogue_csv)

    sources = pd.DataFrame(
        {
            "ra": [214.85, 214.85 + 0.01, 214.5],
            "dec": [52.85, 52.85, 52.5],
        }
    )
    # row 0: exact match, row 1: ~1 arcmin off (outside 1.5"), row 2: far away
    matched = matcher.cross_match(sources, radius_arcsec=1.5)

    assert matched.tolist() == [True, False, False]


def test_cross_match_without_loaded_catalogue_returns_all_false():
    matcher = ContaminantCrossMatch()
    sources = pd.DataFrame({"ra": [1.0, 2.0], "dec": [1.0, 2.0]})

    matched = matcher.cross_match(sources)

    assert matched.tolist() == [False, False]


def test_flag_emission_line_galaxies_detects_lyman_alpha_alias():
    matcher = ContaminantCrossMatch()
    # Lyman-alpha (0.1216um) redshifts into F090W's pivot (~0.902um) at z ~ 6.42
    z_alias = 0.902 / 0.1216 - 1.0
    df = pd.DataFrame({"z_phot": [z_alias, 2.0]})

    flagged = matcher.flag_emission_line_galaxies(df)

    assert flagged.iloc[0] == True  # noqa: E712
    assert flagged.iloc[1] == False  # noqa: E712


# ── apply_quality_pipeline (integration smoke test) ───────────────────────


def test_apply_quality_pipeline_appends_qc_variables_and_promotes_data_level():
    ds, _ = _synthetic_residual_dataset(n=40, seed=4)

    result = apply_quality_pipeline(ds, CONFIG_DIR / "quality_config.yaml")

    for var in (
        "qc_anomaly_score",
        "qc_iso_forest_score",
        "qc_umap_outlier",
        "qc_agn_match",
        "qc_emission_line_flag",
        "qc_eazy_fit_failure",
    ):
        assert var in result.data_vars, f"missing {var}"
        assert result[var].sizes["source_id"] == 40

    assert result.attrs["data_level"] == "b1"
    # No injected chi2_eazy<0 sentinels in this fixture -- nothing excluded.
    assert not result["qc_eazy_fit_failure"].values.any()
    assert np.isfinite(result["qc_anomaly_score"].values).all()


def test_apply_quality_pipeline_excludes_eazy_fit_failures_from_scoring():
    """chi2_eazy < 0 (EAZY's fit-failure sentinel) sources must be flagged via
    qc_eazy_fit_failure and excluded from anomaly scoring -- not just from the
    model fit, but from getting *any* score at all (NaN), so they can never
    dominate a "top outliers" ranking downstream."""
    ds, outlier_idx = _synthetic_residual_dataset(n=60, seed=5, n_outliers=4)

    failed_idx = np.array([0, 1, 2])
    ds["chi2_eazy"].values[failed_idx] = -1.0

    result = apply_quality_pipeline(ds, CONFIG_DIR / "quality_config.yaml")

    failure_flag = result["qc_eazy_fit_failure"].values
    assert failure_flag[failed_idx].all()
    assert not failure_flag[np.setdiff1d(np.arange(60), failed_idx)].any()

    ensemble_score = result["qc_anomaly_score"].values
    iso_score = result["qc_iso_forest_score"].values
    umap_score = result["qc_umap_outlier"].values
    assert np.isnan(ensemble_score[failed_idx]).all()
    assert np.isnan(iso_score[failed_idx]).all()
    assert np.isnan(umap_score[failed_idx]).all()

    non_failed = np.setdiff1d(np.arange(60), failed_idx)
    assert np.isfinite(ensemble_score[non_failed]).all()
    assert np.isfinite(iso_score[non_failed]).all()
    assert np.isfinite(umap_score[non_failed]).all()
