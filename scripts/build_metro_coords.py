#!/usr/bin/env python3
"""Download the U.S. Census 2023 CBSA Gazetteer and write metro centroids
(CBSA code -> latitude, longitude) to ``data/coords/cbsa_centroids.csv``.  These
centroids supply the geographic metric for the spatial-kernel (Conley) standard error
computed by ``scripts/spatial_kernel_se.py``.  Public-domain Census data.

Run from the repo root (needs internet):
    python scripts/build_metro_coords.py
"""
import csv
import io
import os
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
       "2023_Gazetteer/2023_Gaz_cbsa_national.zip")
OUT = os.path.join(ROOT, "data", "coords", "cbsa_centroids.csv")


def _col(row, key):
    """Fetch a column whose header may carry stray surrounding whitespace."""
    for k in row:
        if k.strip().upper() == key:
            return row[k]
    return None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("downloading", URL, flush=True)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    txt = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
    text = zf.read(txt).decode("latin-1")
    rows = csv.DictReader(io.StringIO(text), delimiter="\t")
    n = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cbsa", "lat", "lon"])
        for r in rows:
            cbsa = (_col(r, "CBSAFP") or _col(r, "GEOID") or "").strip()
            lat = (_col(r, "INTPTLAT") or "").strip()
            lon = (_col(r, "INTPTLONG") or "").strip()
            if cbsa and lat and lon:
                w.writerow([cbsa, lat, lon])
                n += 1
    print(f"wrote {n} CBSA centroids -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
