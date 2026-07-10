"""
test_output.py — Unit tests for pipeline.output (CatalogueExporter, DiagnosticPlotter)
=========================================================================================
Builds on the shared conftest.py fixtures (`synthetic_catalogue`, `synthetic_residuals`)
rather than hitting the network or real repo paths. All file output goes to `tmp_path`.

Note: `synthetic_residuals` predates `quality.py`'s final schema — it lacks `ra`/`dec`
and stores its anomaly score as `anomaly_score` (int-typed qc_* flags) rather than the
`qc_anomaly_score` / bool flags `quality.apply_quality_pipeline` actually produces. The
`b1_dataset` fixture below augments a deep copy of the shared fixtures into a proper
b1-shaped Dataset without mutating the shared, session-scoped originals.
"""

import re
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astropy.io import fits

from pipeline.output import (
    ENSEMBLE_SCORE_VAR,
    CatalogueExporter,
    DiagnosticPlotter,
    _fits_column_format,
    _flatten_for_export,
)


@pytest.fixture
def b1_dataset(synthetic_catalogue, synthetic_residuals):
    """A full b1-shaped Dataset built on top of the shared conftest fixtures.

    Returns (ds, injected) where injected["agn_idx"]/["line_idx"] are source_id
    indices deliberately given a high ensemble score *and* qc_agn_match /
    qc_emission_line_flag = True, so tests can verify they get excluded from
    the "unexplained outliers" table.
    """
    ds = synthetic_residuals.copy(deep=True)  # never mutate the shared session fixture
    n = ds.sizes["source_id"]
    rng = np.random.default_rng(99)

    ds["ra"] = xr.DataArray(synthetic_catalogue["ra"].to_numpy()[:n], dims=["source_id"])
    ds["dec"] = xr.DataArray(synthetic_catalogue["dec"].to_numpy()[:n], dims=["source_id"])
    ds["z_a"] = xr.DataArray(ds["z_phot"].values * (1 + rng.normal(0, 0.02, size=n)), dims=["source_id"])
    ds["chi2_eazy"] = xr.DataArray(rng.lognormal(0.5, 0.7, size=n), dims=["source_id"])
    ds["template_id"] = xr.DataArray(rng.integers(1, 13, size=n), dims=["source_id"])

    # A meaningful ensemble score distribution (conftest's `anomaly_score` is an
    # all-zero placeholder), with a genuine top-scoring tail.
    scores = np.clip(rng.beta(2, 6, size=n), 0.0, 1.0)
    top_idx = rng.choice(n, size=25, replace=False)
    scores[top_idx] = rng.uniform(0.82, 0.99, size=len(top_idx))
    ds[ENSEMBLE_SCORE_VAR] = xr.DataArray(scores, dims=["source_id"])
    ds["qc_iso_forest_score"] = xr.DataArray(np.clip(scores + rng.normal(0, 0.05, size=n), 0, 1), dims=["source_id"])
    ds["qc_umap_outlier"] = xr.DataArray((scores > 0.9).astype(float), dims=["source_id"])

    # A handful of the top-scoring sources are "explained" by a contaminant
    # cross-match. Overwrite conftest's placeholder int-typed all-zero flags
    # with proper bool arrays carrying real structure.
    agn_idx = top_idx[:3]
    line_idx = top_idx[3:6]
    agn_flag = np.zeros(n, dtype=bool)
    line_flag = np.zeros(n, dtype=bool)
    agn_flag[agn_idx] = True
    line_flag[line_idx] = True
    ds["qc_agn_match"] = xr.DataArray(agn_flag, dims=["source_id"])
    ds["qc_emission_line_flag"] = xr.DataArray(line_flag, dims=["source_id"])

    # Replace (not merge) attrs: the shared fixture's `"synthetic": True` bool
    # attribute isn't netCDF4-attribute-safe, and isn't part of a real b1
    # Dataset's attrs (produced by quality.apply_quality_pipeline) anyway.
    ds.attrs = {"survey": "ceers", "data_level": "b1", "creation_timestamp": "2026-01-01T00:00:00"}
    return ds, {"agn_idx": agn_idx, "line_idx": line_idx, "top_idx": top_idx}


# ── module-level helpers ────────────────────────────────────────────────


def test_flatten_for_export_excludes_residual_matrix(b1_dataset):
    ds, _ = b1_dataset
    df = _flatten_for_export(ds)

    assert "residuals" not in df.columns
    assert len(df) == ds.sizes["source_id"]
    for col in ("source_id", "ra", "dec", "z_phot", "qc_agn_match", ENSEMBLE_SCORE_VAR):
        assert col in df.columns


def test_fits_column_format_maps_dtypes():
    assert _fits_column_format(pd.Series([True, False])) == "L"
    assert _fits_column_format(pd.Series([1, 2, 3], dtype="int64")) == "K"
    assert _fits_column_format(pd.Series([1.0, 2.5])) == "D"
    assert _fits_column_format(pd.Series(["abc", "de"])).endswith("A")


# ── CatalogueExporter ────────────────────────────────────────────────────


def test_to_fits_writes_header_metadata_and_qc_columns(tmp_path, b1_dataset):
    ds, _ = b1_dataset
    out_path = CatalogueExporter().to_fits(ds, tmp_path / "nested" / "catalogue.fits")

    assert out_path.exists()
    with fits.open(out_path) as hdul:
        header = hdul[0].header
        assert header["SURVEY"] == "ceers"
        assert header["DATALVL"] == "b1"
        assert header["NSOURCES"] == ds.sizes["source_id"]
        assert header["CREATED"] == "2026-01-01T00:00:00"
        assert "PIPEVERS" in header

        table = hdul[1].data
        assert len(table) == ds.sizes["source_id"]
        for col in ("ra", "dec", "z_phot", "qc_agn_match", "qc_emission_line_flag", ENSEMBLE_SCORE_VAR):
            assert col in table.columns.names


def test_to_csv_contains_source_metadata_and_qc_columns(tmp_path, b1_dataset):
    ds, _ = b1_dataset
    out_path = CatalogueExporter().to_csv(ds, tmp_path / "catalogue.csv")

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == ds.sizes["source_id"]
    for col in ("source_id", "ra", "dec", "z_phot", ENSEMBLE_SCORE_VAR, "qc_agn_match"):
        assert col in df.columns
    assert "residuals" not in df.columns


def test_to_netcdf_generates_tsdat_filename_in_directory(tmp_path, b1_dataset):
    ds, _ = b1_dataset
    out_path = CatalogueExporter().to_netcdf(ds, tmp_path)

    assert out_path.parent == tmp_path
    assert re.fullmatch(r"ceers\.b1\.\d{8}\.\d{6}\.nc", out_path.name)
    assert out_path.exists()

    reloaded = xr.open_dataset(out_path)
    try:
        assert reloaded.sizes["source_id"] == ds.sizes["source_id"]
        assert "residuals" in reloaded.data_vars
    finally:
        reloaded.close()


def test_to_netcdf_uses_explicit_file_path_as_is(tmp_path, b1_dataset):
    ds, _ = b1_dataset
    explicit_path = tmp_path / "custom_name.nc"
    out_path = CatalogueExporter().to_netcdf(ds, explicit_path)

    assert out_path == explicit_path
    assert out_path.exists()


# ── DiagnosticPlotter ────────────────────────────────────────────────────


def test_redshift_anomaly_rate_computes_expected_rate_per_bin(tmp_path):
    """Precise regression test: a controlled 10-source, single-bin dataset
    with a known 3/10 flagged fraction and its Poisson error."""
    n = 10
    z = np.full(n, 2.2)  # all fall in the [2.0, 2.5) bin
    scores = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ds = xr.Dataset(
        {"z_phot": (["source_id"], z), ENSEMBLE_SCORE_VAR: (["source_id"], scores)},
        coords={"source_id": np.arange(n)},
        attrs={"survey": "ceers", "data_level": "b1"},
    )

    with mock.patch("matplotlib.axes.Axes.errorbar") as mock_errorbar:
        out_path = DiagnosticPlotter().redshift_anomaly_rate(ds, tmp_path)

    assert out_path == tmp_path / "anomaly_rate_vs_redshift.pdf"
    assert out_path.exists() and out_path.stat().st_size > 0

    args, kwargs = mock_errorbar.call_args
    x, y = args[0], args[1]
    assert len(x) == 1
    np.testing.assert_allclose(y, [0.3])
    np.testing.assert_allclose(kwargs["yerr"], [np.sqrt(3) / 10])


def test_redshift_anomaly_rate_skips_empty_bins(tmp_path):
    """Two well-separated redshift clusters should produce exactly two bins,
    not a run of empty bins in between."""
    z = np.array([1.1, 1.2, 8.6, 8.7])
    scores = np.array([0.9, 0.1, 0.9, 0.9])
    ds = xr.Dataset(
        {"z_phot": (["source_id"], z), ENSEMBLE_SCORE_VAR: (["source_id"], scores)},
        coords={"source_id": np.arange(4)},
        attrs={"survey": "ceers", "data_level": "b1"},
    )

    with mock.patch("matplotlib.axes.Axes.errorbar") as mock_errorbar:
        DiagnosticPlotter().redshift_anomaly_rate(ds, tmp_path)

    args, _ = mock_errorbar.call_args
    assert len(args[0]) == 2  # exactly the two occupied bins, no empty ones in between


def test_sky_distribution_highlights_top_fraction(tmp_path):
    n = 20
    ra = np.arange(n, dtype=float)
    dec = np.arange(n, dtype=float)
    scores = np.linspace(0, 1, n)  # strictly increasing -> top 20% is unambiguous
    ds = xr.Dataset(
        {"ra": (["source_id"], ra), "dec": (["source_id"], dec), ENSEMBLE_SCORE_VAR: (["source_id"], scores)},
        coords={"source_id": np.arange(n)},
        attrs={"survey": "ceers", "data_level": "b1"},
    )

    with mock.patch("matplotlib.axes.Axes.scatter") as mock_scatter:
        out_path = DiagnosticPlotter().sky_distribution(ds, tmp_path, top_fraction=0.2)

    assert out_path == tmp_path / "sky_distribution.pdf"
    assert out_path.exists() and out_path.stat().st_size > 0

    assert mock_scatter.call_count == 2
    highlighted_ra = mock_scatter.call_args_list[1].args[0]
    expected_ra = ra[scores >= np.quantile(scores, 0.8)]
    np.testing.assert_allclose(sorted(highlighted_ra), sorted(expected_ra))


def _parse_latex_table_ids(latex_text: str) -> set:
    body = latex_text.split(r"\midrule")[1].split(r"\bottomrule")[0]
    ids = set()
    for line in body.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        first_field = line.split("&")[0].strip()
        ids.add(int(first_field))
    return ids


def test_score_summary_table_excludes_agn_and_emission_line_matches(tmp_path, b1_dataset):
    ds, injected = b1_dataset
    out_path = DiagnosticPlotter().score_summary_table(ds, tmp_path, n_top=20)

    assert out_path == tmp_path / "top_outliers.tex"
    content = out_path.read_text()
    # table* (not table): 7 numeric columns don't fit a single column in a
    # twocolumn AASTeX document, so score_summary_table spans both columns.
    assert r"\begin{table*}" in content and r"\label{tab:top_outliers}" in content

    ids_in_table = _parse_latex_table_ids(content)
    assert 0 < len(ids_in_table) <= 20
    excluded = set(injected["agn_idx"].tolist()) | set(injected["line_idx"].tolist())
    assert ids_in_table.isdisjoint(excluded)


def test_score_summary_table_handles_int_typed_qc_flags(tmp_path, synthetic_catalogue, synthetic_residuals):
    """Regression test: conftest.py's synthetic_residuals stores qc_agn_match
    / qc_emission_line_flag as int (0/1), not bool. `~` on an int Series is a
    bitwise (not logical) negation, so score_summary_table must cast
    explicitly or this raises/misbehaves.
    """
    ds = synthetic_residuals.copy(deep=True)  # keep the native int-dtype qc_* flags
    n = ds.sizes["source_id"]
    ds["ra"] = xr.DataArray(synthetic_catalogue["ra"].to_numpy()[:n], dims=["source_id"])
    ds["dec"] = xr.DataArray(synthetic_catalogue["dec"].to_numpy()[:n], dims=["source_id"])
    ds[ENSEMBLE_SCORE_VAR] = xr.DataArray(np.linspace(0, 1, n), dims=["source_id"])

    assert ds["qc_agn_match"].dtype.kind in "iu"  # confirms we're exercising the int-dtype path

    out_path = DiagnosticPlotter().score_summary_table(ds, tmp_path)

    assert out_path.exists()
    content = out_path.read_text()
    assert r"\begin{table*}" in content
    ids_in_table = _parse_latex_table_ids(content)
    assert len(ids_in_table) == 20  # all sources are "unexplained" (flags are all-zero/False)
