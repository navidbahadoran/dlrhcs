"""Geographic metric utilities for the spatial-kernel (Conley) standard error.

The empirical panels carry no metric by default, so the headline cross-sectional
standard error is the by-period cluster (sensitivity) form.  When metro centroids are
supplied (``scripts/build_metro_coords.py`` writes them), these helpers turn the
latitude/longitude into the great-circle distance matrix used by
:func:`dlrhcs.onestep.xs_se_geo`, giving the theorem-backed spatial-mixing standard
error with an explicit, credible metric.
"""
import csv

import numpy as np

EARTH_KM = 6371.0088


def haversine_matrix(lat, lon):
    """``N x N`` great-circle distance matrix in kilometres from latitude/longitude
    arrays given in decimal degrees."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_centroids(path, codes):
    """Align a CBSA-centroid table to a panel's unit order.

    ``path`` is a CSV with columns ``cbsa,lat,lon`` (written by
    ``scripts/build_metro_coords.py``).  ``codes`` is the panel's unit order (CBSA
    codes, in the same order as the columns of ``Y``).  Returns ``(lat, lon, matched)``
    where ``matched`` is a boolean mask of the units found in the table; unmatched
    units carry ``NaN`` coordinates and should be dropped by the caller before forming
    the distance matrix.
    """
    table = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            table[str(row["cbsa"]).strip()] = (float(row["lat"]), float(row["lon"]))
    n = len(codes)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    for k, c in enumerate(codes):
        v = table.get(str(c).strip())
        if v is not None:
            lat[k], lon[k] = v
    matched = ~np.isnan(lat)
    return lat, lon, matched
