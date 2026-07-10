"""
test_standardise.py — Unit tests for pipeline.standardise.SEDStandardiser
============================================================================
Uses synthetic DataFrames only; no EAZY installation or network access
required (run_eazy_fit naturally exercises its documented stub fallback
since the `eazy` CLI is not installed in the test environment).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.standardise import (
    BAND_PIVOT_WAVELENGTH_UM,
    SEDStandardiser,
    log_data_level_transition,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"
BANDS = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W"]


@pytest.fixture
def standardiser() -> SEDStandardiser:
    return SEDStandardiser(CONFIG_PATH, survey="ceers")


def _make_catalogue(n=10, rng=None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(0)
    data = {
        "ra": rng.uniform(214.8, 215.0, size=n),
        "dec": rng.uniform(52.8, 53.0, size=n),
        "z_phot": rng.uniform(0.1, 12.0, size=n),  # deliberately includes out-of-range values
        "survey": ["ceers"] * n,
    }
    for band in BANDS:
        flux = rng.lognormal(mean=1.0, sigma=0.5, size=n)
        data[f"{band.lower()}_flux"] = flux
        data[f"{band.lower()}_flux_err"] = flux * 0.1
    return pd.DataFrame(data)


# ── preprocess ────────────────────────────────────────────────────────────


def test_preprocess_filters_out_of_range_redshifts(standardiser):
    df = _make_catalogue(n=20, rng=np.random.default_rng(1))
    # Force a subset outside the configured [0.5, 10.0] window.
    df.loc[0:2, "z_phot"] = [0.1, 0.2, 0.3]
    df.loc[3:5, "z_phot"] = [10.5, 11.0, 11.5]

    result = standardiser.preprocess(df)

    z_min, z_max = standardiser.redshift_range
    assert (result["z_phot"].between(z_min, z_max)).all()
    assert len(result) < len(df)


def test_preprocess_drops_sources_with_insufficient_band_coverage(standardiser):
    df = _make_catalogue(n=10, rng=np.random.default_rng(2))
    df.loc[:, "z_phot"] = 2.0  # keep every source inside the redshift window

    # Source 0 only has 2 valid (non-null, positive) flux measurements.
    for band in BANDS:
        df.loc[0, f"{band.lower()}_flux"] = np.nan
    df.loc[0, "f090w_flux"] = 1.0
    df.loc[0, "f115w_flux"] = 1.0

    result = standardiser.preprocess(df)

    assert len(result) == len(df) - 1


def test_preprocess_returns_dataframe_with_reset_index(standardiser):
    df = _make_catalogue(n=5, rng=np.random.default_rng(3))
    df["z_phot"] = 2.0

    result = standardiser.preprocess(df)

    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == list(range(len(result)))


@pytest.mark.parametrize("ceers_col", ["lp_z_best", "lp_z_med"])
def test_preprocess_renames_ceers_lephare_redshift_columns_to_z_phot(standardiser, ceers_col):
    """CEERS DR1.0 (Cox et al. 2025) names its LePHARE photo-z columns
    lp_z_best/lp_z_med rather than z_phot; preprocess must recognise and
    rename them so downstream stages (run_eazy_fit, extract_residuals) work
    unchanged."""
    df = _make_catalogue(n=5, rng=np.random.default_rng(5))
    df = df.rename(columns={"z_phot": ceers_col})
    df[ceers_col] = 2.0

    result = standardiser.preprocess(df)

    assert "z_phot" in result.columns
    assert ceers_col not in result.columns
    assert (result["z_phot"] == 2.0).all()


@pytest.mark.parametrize("err_suffix", ["fluxerr_emp", "fluxerr_se"])
def test_preprocess_canonicalises_ceers_flux_err_columns(standardiser, err_suffix):
    """CEERS DR1.0 (Cox et al. 2025) names its per-band error columns
    <band>_fluxerr_emp/_se rather than <band>_flux_err. Without canonicalising
    these to <band>_flux_err, every per-band error lookup in extract_residuals
    silently returns NaN, no source ever has >=2 usable bands, and the
    residual matrix collapses to all-NaN -- verified against the real CEERS
    catalogue, where this collapsed every downstream anomaly score to 0."""
    df = _make_catalogue(n=5, rng=np.random.default_rng(6))
    df["z_phot"] = 2.0
    rename = {f"{b.lower()}_flux_err": f"{b.lower()}_{err_suffix}" for b in BANDS}
    df = df.rename(columns=rename)

    result = standardiser.preprocess(df)

    # standardiser is fixtured with survey="ceers", which configures
    # raw_flux_unit: nJy (see pipeline_config.yaml) -- un-suffixed flux/
    # flux_err columns are scaled by 1e-3 on the way to uJy.
    for b in BANDS:
        assert f"{b.lower()}_flux_err" in result.columns
        assert f"{b.lower()}_{err_suffix}" not in result.columns
        assert np.allclose(result[f"{b.lower()}_flux_err"], df[f"{b.lower()}_{err_suffix}"] * 1e-3)


# ── extract_residuals ────────────────────────────────────────────────────


def test_extract_residuals_shape_and_dims(standardiser):
    df = _make_catalogue(n=6, rng=np.random.default_rng(4))
    df["z_phot"] = np.linspace(0.5, 5.0, 6)
    fit_df = pd.DataFrame(
        {
            "id": np.arange(1, 7),
            "z_a": df["z_phot"] * 1.01,
            "chi2": np.full(6, 1.5),
            "template_id": np.full(6, 3),
        }
    )

    ds = standardiser.extract_residuals(df, fit_df)

    assert set(ds.dims) == {"source_id", "band"}
    assert ds.sizes["source_id"] == 6
    assert ds.sizes["band"] == len(standardiser._flux_bands_present(df))
    assert "residuals" in ds.data_vars
    assert ds["residuals"].shape == (6, ds.sizes["band"])


def test_extract_residuals_attaches_required_metadata(standardiser):
    df = _make_catalogue(n=4, rng=np.random.default_rng(5))
    df["z_phot"] = 2.0
    fit_df = pd.DataFrame(
        {
            "id": np.arange(1, 5),
            "z_a": [2.0, 2.0, 2.0, 2.0],
            "chi2": [1.0, 1.2, 0.9, 1.1],
            "template_id": [1, 2, 3, 4],
        }
    )

    ds = standardiser.extract_residuals(df, fit_df)

    for key in ("survey", "data_level", "creation_timestamp", "eazy_version", "n_sources", "n_bands"):
        assert key in ds.attrs, f"missing global attribute: {key}"
    assert ds.attrs["data_level"] == "a1"
    assert ds.attrs["survey"] == "ceers"
    assert ds.attrs["n_sources"] == 4


def test_extract_residuals_raises_on_length_mismatch(standardiser):
    df = _make_catalogue(n=4, rng=np.random.default_rng(6))
    df["z_phot"] = 2.0
    fit_df = pd.DataFrame(
        {"id": [1, 2], "z_a": [2.0, 2.0], "chi2": [1.0, 1.0], "template_id": [1, 1]}
    )

    with pytest.raises(ValueError):
        standardiser.extract_residuals(df, fit_df)


def test_band_pivot_wavelengths_cover_configured_bands(standardiser):
    for band in standardiser.band_list:
        assert band in BAND_PIVOT_WAVELENGTH_UM


# ── run_eazy_fit ────────────────────────────────────────────────────────
#
# `run_eazy_fit` tries a real `eazy-py` fit first and only falls back to
# the synthetic stub if that raises. The fallback test below forces that
# path with monkeypatch rather than relying on eazy-py being absent from
# the test environment (it is installed here, with real gbrammer/
# eazy-photoz templates -- see `SEDStandardiser._ensure_eazy_templates`).


def test_run_eazy_fit_falls_back_to_stub_on_fit_failure(standardiser, tmp_path, monkeypatch):
    def _boom(self, df, ids, output_dir):
        raise RuntimeError("simulated eazy-py failure")

    monkeypatch.setattr(SEDStandardiser, "_fit_with_eazy_py", _boom)

    df = _make_catalogue(n=5, rng=np.random.default_rng(7))
    df["z_phot"] = 2.0

    result = standardiser.run_eazy_fit(df, tmp_path)

    assert {"z_a", "chi2", "template_id"}.issubset(result.columns)
    assert (result["chi2"] > 0).all()  # lognormal domain
    assert len(result) == len(df)
    assert standardiser._eazy_version == "synthetic-stub"
    assert standardiser._last_fit_products is None


def test_run_eazy_fit_uses_real_eazy_py(standardiser, tmp_path):
    """End-to-end: a real eazy-py fit against gbrammer/eazy-photoz templates
    produces (z_a, chi2, template_id) and stashes best-fit template
    photometry for extract_residuals to use for real per-band residuals."""
    df = _make_catalogue(n=5, rng=np.random.default_rng(7))
    df["z_phot"] = 2.0

    result = standardiser.run_eazy_fit(df, tmp_path)

    assert {"z_a", "chi2", "template_id"}.issubset(result.columns)
    assert len(result) == len(df)
    assert standardiser._eazy_version.startswith("eazy-py ")
    assert standardiser._last_fit_products is not None
    assert standardiser._last_fit_products["n"] == len(df)
    assert standardiser._last_fit_products["fmodel"].shape == (
        len(df),
        len(standardiser._flux_bands_present(df)),
    )

    ds = standardiser.extract_residuals(df, result)
    assert ds.attrs["residual_model"] == "eazy_best_fit_template"


# ── module-level helper ──────────────────────────────────────────────────


def test_log_data_level_transition_does_not_raise(caplog):
    with caplog.at_level("INFO"):
        log_data_level_transition("raw", "a1", 100, 80)
    assert any("raw -> a1" in record.message for record in caplog.records)
