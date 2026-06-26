#!/usr/bin/env python3
"""Render the two heterogeneity choropleths (real US-state geography) to PDF for the
paper, from the empirical JSON outputs.  Uses Plotly's built-in USA-states geometry
(no shapefile) and a continuous colour scale, so the tight unemployment range shows a
proper gradient.  Writes paper/fig_emp_map_housing.pdf and paper/fig_emp_map_unemp.pdf.

Requires plotly + kaleido + a local Chrome/Chromium (Kaleido renders via headless
Chrome).  Run from the repo root after the empirical run:
    python scripts/make_maps.py
"""
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMP = os.path.join(ROOT, "outputs", "empirical")
COV = os.path.join(ROOT, "data", "zillow", "metro_monthly_covariates_2000_present.csv")
PAPER = os.path.join(ROOT, "paper")
STATES = set("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
             "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV "
             "WI WY".split())


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


def choropleth(data, title, scale, cbar_title, path):
    st = sorted(data)
    fig = go.Figure(go.Choropleth(
        locations=st, locationmode="USA-states", z=[data[s] for s in st],
        colorscale=scale, marker_line_color="white", marker_line_width=0.5,
        colorbar=dict(title=cbar_title, thickness=14, len=0.85),
        zmin=min(data.values()), zmax=max(data.values())))
    fig.update_layout(geo=dict(scope="usa", lakecolor="white", bgcolor="rgba(0,0,0,0)"),
                      title=dict(text=title, x=0.5, font=dict(size=15)),
                      margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
    fig.write_image(path, format="pdf", width=760, height=470)
    print("wrote", path)


def main():
    zillow, unemp = state_means()
    print(f"housing states={len(zillow)} range=[{min(zillow.values()):.3f},{max(zillow.values()):.3f}]")
    print(f"unemp states={len(unemp)} range=[{min(unemp.values()):.3f},{max(unemp.values()):.3f}]")
    choropleth(zillow, "House-price momentum: cumulative persistence a+b by state",
               "Blues", "a+b", os.path.join(PAPER, "fig_emp_map_housing.pdf"))
    choropleth(unemp, "Idiosyncratic unemployment persistence: lag-1 a by state",
               "Tealgrn", "a", os.path.join(PAPER, "fig_emp_map_unemp.pdf"))


if __name__ == "__main__":
    main()
