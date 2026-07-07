"""
conftest.py — Shared pytest fixtures for jwst-sed-anomaly
==========================================================
All fixtures are purely synthetic — no network calls, no real data.
This allows the full test suite to run in CI without MAST access.

Fixtures
--------
synthetic_catalogue   : 500-row DataFrame with realistic JWST photometry
synthetic_residuals   : xarray Dataset of per-band SED residuals
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime

# Canonical band list mirroring NIRCam + MIRI photometry used in CEERS/JADES
BANDS = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F410M", "F444W", "F770W"]
N_SOURCES = 500
RNG = np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_catalogue() -> pd.DataFrame:
    """
    500-row DataFrame simulating a JWST photometric catalogue.

    Columns
    -------
    ra, dec          : sky coordinates (degrees)
    z_phot           : photometric redshift (lognormal around z~2)
    survey           : source survey label
    F*W_flux         : observed flux in µJy (lognormal)
    F*W_flux_err     : flux uncertainty ~5–15% of flux
    """
    n = N_SOURCES

    # Sky coordinates: a mock ~10 arcmin field
    ra  = RNG.uniform(214.8, 215.0, size=n)
    dec = RNG.uniform(52.8,  53.0,  size=n)

    # Photo-z: lognormal peaked around z~2, range [0.3, 9]
    z_phot = np.clip(RNG.lognormal(mean=0.7, sigma=0.5, size=n), 0.3, 9.5)

    data: dict = {
        "ra":     ra,
        "dec":    dec,
        "z_phot": z_phot,
        "survey": ["ceers"] * n,
    }

    # Flux columns: lognormal in µJy, errors ~10% of flux
    for band in BANDS:
        flux = RNG.lognormal(mean=1.5, sigma=0.8, size=n)   # µJy
        err  = flux * RNG.uniform(0.05, 0.15, size=n)
        data[f"{band}_flux"]     = flux
        data[f"{band}_flux_err"] = err

    # Inject ~2% anomalous sources (10x excess in one band)
    anomaly_idx = RNG.choice(n, size=int(0.02 * n), replace=False)
    for idx in anomaly_idx:
        spike_band = RNG.choice(BANDS)
        data[f"{spike_band}_flux"][idx] *= 10.0

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def synthetic_residuals(synthetic_catalogue: pd.DataFrame) -> xr.Dataset:
    """
    xarray Dataset of synthetic per-band SED residuals.

    Simulates the output of SEDStandardiser.extract_residuals():
    residuals are (obs - model) / err, mostly ~ N(0,1) with a few large
    outliers matching the anomalous sources injected in synthetic_catalogue.

    Also pre-populates qc_* flag variables as expected by the quality stage.
    """
    df = synthetic_catalogue
    n  = len(df)

    residuals = RNG.standard_normal((n, len(BANDS)))

    # Inject large residuals for sources that had flux spikes
    # (we don't track exact indices here, so inject randomly at ~2%)
    spike_mask = RNG.random(size=(n, len(BANDS))) < 0.02
    residuals[spike_mask] *= RNG.uniform(5, 15, size=spike_mask.sum())

    ds = xr.Dataset(
        {
            "residuals": xr.DataArray(
                residuals,
                dims=["source_id", "band"],
                attrs={"units": "sigma", "description": "Normalised SED residuals (obs-model)/err"},
            ),
            # QC flags (0 = pass, 1 = flagged)
            "qc_residual_outlier":      xr.DataArray(np.zeros(n, dtype=int), dims=["source_id"]),
            "qc_poor_coverage":         xr.DataArray(np.zeros(n, dtype=int), dims=["source_id"]),
            "qc_chi2_excess":           xr.DataArray(np.zeros(n, dtype=int), dims=["source_id"]),
            "qc_redshift_out_of_range": xr.DataArray(np.zeros(n, dtype=int), dims=["source_id"]),
            "qc_agn_match":             xr.DataArray(np.zeros(n, dtype=int), dims=["source_id"]),
            "qc_emission_line_flag":    xr.DataArray(np.zeros(n, dtype=int), dims=["source_id"]),
            "anomaly_score":            xr.DataArray(np.zeros(n, dtype=float), dims=["source_id"]),
            "z_phot":                   xr.DataArray(df["z_phot"].values, dims=["source_id"]),
        },
        coords={
            "source_id": np.arange(n),
            "band":      BANDS,
        },
        attrs={
            "survey":             "ceers",
            "data_level":         "a1",
            "creation_timestamp": datetime.utcnow().isoformat(),
            "n_sources":          n,
            "n_bands":            len(BANDS),
            "synthetic":          True,
        },
    )

    return ds
