"""
test_retriever.py — Unit tests for pipeline.retriever.MASTRetriever
=====================================================================
All tests use mocks/synthetic files; no real network calls are made.
"""

from pathlib import Path

import pandas as pd
import pytest
from astropy.table import Table

from pipeline.retriever import (
    BAND_PREFIXES,
    CEERS_DR1_URL,
    REQUIRED_COLUMNS,
    MASTRetriever,
)


@pytest.fixture
def retriever(tmp_path) -> MASTRetriever:
    return MASTRetriever(raw_dir=tmp_path)


# ── fetch_* fallback behaviour ────────────────────────────────────────────


def test_fetch_ceers_falls_back_to_https_when_astroquery_fails(retriever, tmp_path, mocker):
    """If the astroquery.mast path raises, fetch_ceers must fall back to the
    documented HTTPS download rather than propagating the error."""
    mock_mast = mocker.patch.object(
        MASTRetriever, "_fetch_ceers_via_mast", side_effect=RuntimeError("no astroquery access")
    )
    mock_https = mocker.patch.object(MASTRetriever, "_download_via_https")

    out_path = tmp_path / "ceers_dr1.fits"
    result = retriever.fetch_ceers(out_path)

    assert result == out_path
    mock_mast.assert_called_once()
    mock_https.assert_called_once()
    called_url = mock_https.call_args.args[0]
    assert called_url == CEERS_DR1_URL


def test_fetch_jades_uses_astroquery_when_available(retriever, tmp_path, mocker):
    """If the astroquery.mast path succeeds, no HTTPS fallback should occur."""
    mock_mast = mocker.patch.object(MASTRetriever, "_fetch_jades_via_mast", return_value=None)
    mock_https = mocker.patch.object(MASTRetriever, "_download_via_https")

    out_path = tmp_path / "jades_dr1.fits"
    result = retriever.fetch_jades(out_path)

    assert result == out_path
    mock_mast.assert_called_once()
    mock_https.assert_not_called()


def test_download_via_https_never_called_with_real_network(retriever, tmp_path, mocker):
    """Sanity check that requests.get is fully mocked and never hits the network.

    CEERS_DR1_URL is a .gz file, so the dummy payload must itself be valid
    gzip data for _download_via_https's transparent decompression to succeed.
    """
    import gzip as gzip_module

    dummy_content = b"dummy bytes"
    mock_get = mocker.patch("pipeline.retriever.requests.get")
    mock_response = mocker.Mock()
    mock_response.iter_content.return_value = [gzip_module.compress(dummy_content)]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    mocker.patch.object(
        MASTRetriever, "_fetch_ceers_via_mast", side_effect=RuntimeError("no astroquery")
    )

    out_path = tmp_path / "ceers_dr1.fits"
    retriever.fetch_ceers(out_path)

    mock_get.assert_called_once()
    assert out_path.read_bytes() == dummy_content


# ── load_catalogue ────────────────────────────────────────────────────────


def test_load_catalogue_standardises_columns_and_adds_survey(retriever, tmp_path):
    csv_path = tmp_path / "ceers_dr1.csv"
    raw = pd.DataFrame(
        {
            "RA": [214.9, 214.95],
            "DEC": [52.9, 52.95],
            "Z_PHOT": [1.2, 3.4],
            "F090W FLUX": [1.1, 2.2],
            "F150W-Flux": [3.3, 4.4],
        }
    )
    raw.to_csv(csv_path, index=False)

    df = retriever.load_catalogue(csv_path)

    assert list(df.columns) == ["ra", "dec", "z_phot", "f090w_flux", "f150w_flux", "survey"]
    assert (df["survey"] == "ceers").all()


def test_load_catalogue_reads_fits_and_infers_survey_from_filename(retriever, tmp_path):
    fits_path = tmp_path / "jades_dr1.fits"
    table = Table({"RA": [10.0], "DEC": [20.0], "Redshift": [2.5]})
    table.write(fits_path, format="fits")

    df = retriever.load_catalogue(fits_path)

    assert set(["ra", "dec", "redshift", "survey"]).issubset(df.columns)
    assert df["survey"].iloc[0] == "jades"


def test_load_catalogue_rejects_unsupported_format(retriever, tmp_path):
    bad_path = tmp_path / "catalogue.json"
    bad_path.write_text("{}")

    with pytest.raises(ValueError):
        retriever.load_catalogue(bad_path)


# ── validate_schema ───────────────────────────────────────────────────────


def test_validate_schema_valid_catalogue(retriever):
    data = {"ra": [1.0, 2.0, 3.0], "dec": [1.0, 2.0, 3.0], "z_phot": [0.5, 1.5, 2.5]}
    for band in BAND_PREFIXES[:4]:
        data[f"{band.lower()}_flux"] = [1.0, 2.0, 3.0]
    df = pd.DataFrame(data)

    report = retriever.validate_schema(df)

    assert report["valid"] is True
    assert report["missing_columns"] == []
    assert report["n_sources"] == 3
    assert report["redshift_range"] == (0.5, 2.5)


def test_validate_schema_flags_missing_columns_and_insufficient_bands(retriever):
    # Missing 'dec', missing redshift/z_phot, and only 2 flux bands present.
    df = pd.DataFrame(
        {
            "ra": [1.0, 2.0],
            "f090w_flux": [1.0, 2.0],
            "f150w_flux": [1.0, 2.0],
        }
    )

    report = retriever.validate_schema(df)

    assert report["valid"] is False
    assert "dec" in report["missing_columns"]
    assert any("redshift" in m for m in report["missing_columns"])
    assert any("photometric_flux_columns" in m for m in report["missing_columns"])
    assert report["redshift_range"] == (None, None)


def test_required_columns_and_band_prefixes_are_module_level_lists():
    assert isinstance(REQUIRED_COLUMNS, list)
    assert "ra" in REQUIRED_COLUMNS and "dec" in REQUIRED_COLUMNS
    assert isinstance(BAND_PREFIXES, list)
    assert len(BAND_PREFIXES) >= 4
