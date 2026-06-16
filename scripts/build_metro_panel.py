#!/usr/bin/env python3
"""Build the clean metro-unemployment panel from the raw BLS LAUS flat file.

This is the data-preparation step for the OPTIONAL second empirical application
(metro-area unemployment), which is implemented but not part of the default run.

Place the raw downloads in a local (git-ignored) ``data/metro/`` folder:
    data/metro/la.data.60.Metro.txt   (all U.S. metropolitan areas)
    data/metro/la.series              (series_id -> area title)
then run:  python scripts/build_metro_panel.py
which writes:
    data/metro/metro_unemployment.csv (YEAR + one column per MSA)

Cleaning steps (see EMPIRICAL_FINDINGS.md):
  * Keep metropolitan statistical areas (area type "B"), the not-seasonally-
    adjusted unemployment-RATE series (LAUMT...3, measure code 03).
  * Use the agency's own ANNUAL AVERAGE (period M13) -- the within-year mean of
    the twelve monthly rates -- so the series carry no seasonal cycle.
  * Keep the balanced set of metros observed in every complete year 1990-2025.

Per-series stationarization (ADF) and standardization happen downstream in
dlrhcs.empirical.load_metro.

Source: U.S. Bureau of Labor Statistics, Local Area Unemployment Statistics,
https://download.bls.gov/pub/time.series/la/  (public domain).
"""
import csv
import hashlib
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRO = os.path.join(ROOT, "data", "metro")
RAW = os.path.join(METRO, "la.data.60.Metro.txt")
SER = os.path.join(METRO, "la.series")
OUT = os.path.join(METRO, "metro_unemployment.csv")
YEARS = list(range(1990, 2026))


def main():
    name = {}
    with open(SER) as fh:
        r = csv.reader(fh, delimiter="\t")
        next(r)
        for row in r:
            # metro statistical areas carry area_type_code "B", measure 03 = rate
            if len(row) >= 7 and row[1].strip() == "B" and row[3].strip() == "03":
                t = row[6].strip().replace("Unemployment Rate:", "").strip()
                t = re.sub(r"\s*Metropolitan Statistical Area\s*\(U\)\s*$", "", t)
                t = re.sub(r"\s*\(U\)\s*$", "", t).strip()
                name[row[0].strip()] = t

    data = {}
    with open(RAW) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            sid = p[0].strip()
            if not sid.startswith("LAUMT") or not sid.endswith("3"):
                continue
            if p[2].strip() != "M13":
                continue
            try:
                data.setdefault(sid, {})[int(p[1])] = float(p[3])
            except ValueError:
                continue

    bal = sorted(s for s in data if all(y in data[s] for y in YEARS))
    print(f"metros total={len(data)}  balanced 1990-2025={len(bal)}")

    seen, cols = {}, []
    for s in bal:
        nm = name.get(s, s)
        k = seen.get(nm, 0)
        seen[nm] = k + 1
        cols.append((s, nm if k == 0 else f"{nm} #{k + 1}"))

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["YEAR"] + [c[1] for c in cols])
        for y in YEARS:
            w.writerow([y] + [f"{data[s][y]:.1f}" for s, _ in cols])

    h = hashlib.sha256()
    with open(RAW, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 16), b""):
            h.update(ch)
    print(f"wrote {OUT}  shape {len(YEARS)} x {len(cols)}")
    print(f"raw la.data.60.Metro sha256[:16] = {h.hexdigest()[:16]}")


if __name__ == "__main__":
    main()
