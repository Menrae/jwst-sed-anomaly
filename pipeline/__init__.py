"""
jwst-sed-anomaly pipeline
=========================
A tsdat-inspired four-stage pipeline for detecting anomalous spectral
energy distributions (SEDs) in high-redshift galaxies using archival
JWST photometric catalogues.

Stages
------
1. Retriever   (retriever.py)   — MAST catalogue ingestion
2. Standardise (standardise.py) — SED fitting + residual extraction → a1 NetCDF
3. Quality     (quality.py)     — anomaly detection + contaminant QC → b1 NetCDF
4. Output      (output.py)      — FITS catalogue + figures + LaTeX table

UW Astronomy · Summer 2026
"""

__version__ = "0.1.0"
__author__ = "Aasha"
__project__ = "JWST SED Anomaly Detection"
