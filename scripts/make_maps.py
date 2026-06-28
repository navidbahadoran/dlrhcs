#!/usr/bin/env python3
"""Render the two heterogeneity choropleths to PDF using matplotlib ONLY -- no Plotly,
no Kaleido, no headless browser.  Downloads a small US-states GeoJSON once (cached at
data/us_states_geo.json; commit it for full offline reproducibility), projects it with
the Albers Equal-Area Conic projection, and shades each state by its estimated dynamic
persistence.  Alaska and Hawaii are omitted for a clean lower-48 map (their values are
in the data).  Writes paper/fig_emp_map_housing.pdf and paper/fig_emp_map_unemp.pdf.

Run from the repo root after the empirical run:  python scripts/make_maps.py
"""
import csv
import json
import math
import os
import re
import urllib.request
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMP = os.path.join(ROOT, "outputs", "empirical")
COV = os.path.join(ROOT, "data", "zillow", "metro_monthly_covariates_2000_present.csv")
PAPER = os.path.join(ROOT, "paper", "figures")
GEO_PATH = os.path.join(ROOT, "data", "us_states_geo.json")
GEO_URL = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
           "master/data/geojson/us-states.json")
ABBR2NAME = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming"}
NAME2ABBR = {v: k for k, v in ABBR2NAME.items()}
STATES = set(ABBR2NAME)


def _state(name):
    for t in re.findall(r"[A-Z]{2}", name):
        if t in STATES:
            return t
    return None


def _cbsa_to_state():
    rows = list(csv.reader(open(COV)))
    h = {c: i for i, c in enumerate(rows[0])}
    out = {}
    for r in rows[1:]:
        s = _state(r[h["cbsa_name"]])
        if s:
            out[r[h["cbsa_code"]].lstrip("0")] = s
    return out


def state_means():
    z = json.load(open(os.path.join(EMP, "zillow_abc.json")))["A"]
    zst = defaultdict(list)
    for c, reg in zip(z["derived"]["coef_by_unit"]["cum_i"], z["regions"]):
        s = _state(reg)
        if s:
            zst[s].append(c)
    c2s = _cbsa_to_state()
    u = json.load(open(os.path.join(EMP, "unemp_abc.json")))["B"]
    ust = defaultdict(list)
    for c, code in zip(u["derived"]["coef_by_unit"]["cum_i"], u["ces"]):
        s = c2s.get(str(code).lstrip("0"))
        if s:
            ust[s].append(c)
    return ({k: float(np.mean(v)) for k, v in zst.items()},
            {k: float(np.mean(v)) for k, v in ust.items()})


def _albers(lon, lat):
    lat, lon = math.radians(lat), math.radians(lon)
    lat0, lon0 = math.radians(23.0), math.radians(-96.0)
    p1, p2 = math.radians(29.5), math.radians(45.5)
    n = (math.sin(p1) + math.sin(p2)) / 2.0
    C = math.cos(p1) ** 2 + 2 * n * math.sin(p1)
    rho0 = math.sqrt(max(C - 2 * n * math.sin(lat0), 0)) / n
    theta = n * (lon - lon0)
    rho = math.sqrt(max(C - 2 * n * math.sin(lat), 0)) / n
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


def load_geojson():
    if not os.path.exists(GEO_PATH):
        print("downloading US-states GeoJSON (one time) ...")
        urllib.request.urlretrieve(GEO_URL, GEO_PATH)
    return json.load(open(GEO_PATH))


def _rings(geom):
    t, coords = geom["type"], geom["coordinates"]
    if t == "Polygon":
        return [coords[0]]
    if t == "MultiPolygon":
        return [poly[0] for poly in coords]
    return []


def choropleth(data, title, cmap_name, cbar_title, path, gj):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    cmap = matplotlib.colormaps[cmap_name]
    norm = Normalize(vmin=min(data.values()), vmax=max(data.values()))
    patches, colors = [], []
    for feat in gj["features"]:
        ab = NAME2ABBR.get(feat["properties"].get("name", ""))
        if ab is None or ab in ("AK", "HI"):
            continue
        v = data.get(ab)
        col = cmap(norm(v)) if v is not None else (0.92, 0.92, 0.92, 1.0)
        for ring in _rings(feat["geometry"]):
            patches.append(Polygon([_albers(x, y) for x, y in ring], closed=True))
            colors.append(col)
    ax.add_collection(PatchCollection(patches, facecolors=colors,
                                      edgecolors="white", linewidths=0.4))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(cbar_title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(PAPER, exist_ok=True)
    zillow, unemp = state_means()
    print(f"housing states={len(zillow)} range=[{min(zillow.values()):.3f},{max(zillow.values()):.3f}]")
    print(f"unemp states={len(unemp)} range=[{min(unemp.values()):.3f},{max(unemp.values()):.3f}]")
    gj = load_geojson()
    choropleth(zillow, "House-price momentum: cumulative persistence a+b by state",
               "Blues", "a + b", os.path.join(PAPER, "fig_emp_map_housing.pdf"), gj)
    choropleth(unemp, "Idiosyncratic unemployment persistence: lag-1 a by state",
               "Greens", "a", os.path.join(PAPER, "fig_emp_map_unemp.pdf"), gj)


if __name__ == "__main__":
    main()
