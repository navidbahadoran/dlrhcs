"""
zillow_covariates_metro_county_aggregate.py

Purpose
-------
Create metropolitan / CBSA covariates for a Zillow metro panel.

This version avoids BEA's GeoFips="MSA" problem by:
1. Downloading monthly metro/CBSA building permits directly from Census BPS.
2. Downloading county GDP from BEA and aggregating counties to CBSA/MSA.
3. Downloading county population from BEA and aggregating counties to CBSA/MSA.
4. Converting annual GDP/population to monthly variables.
5. Merging everything into one monthly metro covariate file.

Why this version?
-----------------
BEA may reject direct MSA calls such as:
    TableName=CAGDP9, GeoFips=MSA

So this script uses:
    TableName=CAGDP9, GeoFips=COUNTY

then aggregates counties to MSA/CBSA using the Census 2023 delineation file.

Required packages
-----------------
pip install pandas requests openpyxl

BEA API key
-----------
Recommended in Git Bash / MINGW64:
    export BEA_API_KEY="YOUR_REAL_BEA_KEY"
    python zillow-covariate.py

Or paste your key directly below:
    BEA_KEY = "YOUR_REAL_BEA_KEY"

Output
------
metro_monthly_covariates_2000_present.csv
"""

import os
import re
import time
import json
import requests
import pandas as pd
from io import BytesIO
from urllib.parse import urljoin


# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 2000

# For Zillow Metro data, usually keep only metropolitan areas, not micropolitan areas.
KEEP_METROPOLITAN_ONLY = True

# BEA key.
# Option A: recommended, use environment variable:
#     export BEA_API_KEY="YOUR_REAL_BEA_KEY"
# Option B: paste directly:
#     BEA_KEY = "YOUR_REAL_BEA_KEY"
BEA_KEY = "0916061D-9348-45F8-938A-0548BF03F198"

# If you want to paste the key directly, uncomment and edit this line:
# BEA_KEY = "YOUR_REAL_BEA_KEY"

# Census 2023 CBSA-to-county delineation file.
# This maps county FIPS to CBSA/MSA code.
CBSA_DELINEATION_URL = (
    "https://www2.census.gov/programs-surveys/metro-micro/"
    "geographies/reference-files/2023/delineation-files/list1_2023.xlsx"
)

# Census BPS cache folder
BPS_CACHE_DIR = "bps_metro_cache"
os.makedirs(BPS_CACHE_DIR, exist_ok=True)

# Output files
OUT_CROSSWALK = "cbsa_county_crosswalk_2023.csv"
OUT_BPS_MONTHLY = "metro_bps_monthly_2000_present.csv"

OUT_COUNTY_GDP_ANNUAL = "bea_county_real_gdp_annual_2001_present.csv"
OUT_MSA_GDP_ANNUAL = "bea_msa_real_gdp_annual_2001_present.csv"
OUT_MSA_GDP_MONTHLY = "bea_msa_real_gdp_monthly_lagged.csv"

OUT_COUNTY_POP_ANNUAL = "bea_county_population_annual_2000_present.csv"
OUT_MSA_POP_ANNUAL = "bea_msa_population_annual_2000_present.csv"
OUT_MSA_POP_MONTHLY = "bea_msa_population_monthly_interpolated.csv"

OUT_FINAL = "metro_monthly_covariates_2000_present.csv"


# ============================================================
# SMALL HELPERS
# ============================================================

def zfill_clean(x, width):
    return (
        x.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(width)
    )


def download_bytes_with_retry(url, tries=5, timeout=60):
    last_error = None

    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{tries} failed: {e}")
            time.sleep(3 * attempt)

    raise RuntimeError(f"Failed after {tries} tries: {url}") from last_error


def find_col_contains(df, required_words):
    """
    Find a column whose name contains all words in required_words, case-insensitive.
    """
    required_words = [w.lower() for w in required_words]

    for col in df.columns:
        low = str(col).lower()
        if all(w in low for w in required_words):
            return col

    raise KeyError(
        f"Could not find column containing {required_words}. "
        f"Available columns: {list(df.columns)}"
    )


def find_col_case_insensitive(df, target):
    target_lower = target.lower()

    for col in df.columns:
        if str(col).lower() == target_lower:
            return col

    raise KeyError(f"Column like {target} not found. Available: {list(df.columns)}")


# ============================================================
# CBSA COUNTY CROSSWALK
# ============================================================

def download_cbsa_county_crosswalk():
    """
    Download Census 2023 CBSA delineation file and create county-to-CBSA crosswalk.

    Output columns:
        cbsa_code
        cbsa_name
        metro_micro_type
        county_name
        state_name
        county_fips
    """
    if os.path.exists(OUT_CROSSWALK):
        print(f"Loading existing file: {OUT_CROSSWALK}")
        return pd.read_csv(
            OUT_CROSSWALK,
            dtype={"cbsa_code": str, "county_fips": str}
        )

    print("Downloading Census CBSA county delineation file...")
    content = download_bytes_with_retry(CBSA_DELINEATION_URL, timeout=120)

    # First read without headers to find the real header row.
    raw = pd.read_excel(BytesIO(content), header=None, engine="openpyxl")

    header_row = None
    for i in range(min(20, len(raw))):
        row_values = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
        if any(v == "cbsa code" for v in row_values):
            header_row = i
            break

    if header_row is None:
        raise RuntimeError("Could not find header row containing 'CBSA Code'.")

    cw = pd.read_excel(BytesIO(content), header=header_row, engine="openpyxl")
    cw.columns = [str(c).strip() for c in cw.columns]

    cbsa_col = find_col_contains(cw, ["CBSA", "Code"])
    cbsa_name_col = find_col_contains(cw, ["CBSA", "Title"])
    type_col = find_col_contains(cw, ["Metropolitan", "Micropolitan"])
    county_col = find_col_contains(cw, ["County"])
    state_name_col = find_col_contains(cw, ["State", "Name"])
    state_fips_col = find_col_contains(cw, ["FIPS", "State"])
    county_fips_col = find_col_contains(cw, ["FIPS", "County"])

    out = cw[[
        cbsa_col,
        cbsa_name_col,
        type_col,
        county_col,
        state_name_col,
        state_fips_col,
        county_fips_col,
    ]].copy()

    out = out.rename(columns={
        cbsa_col: "cbsa_code",
        cbsa_name_col: "cbsa_name",
        type_col: "metro_micro_type",
        county_col: "county_name",
        state_name_col: "state_name",
        state_fips_col: "state_fips",
        county_fips_col: "county_fips_part",
    })

    out = out.dropna(subset=["cbsa_code", "state_fips", "county_fips_part"])

    out["cbsa_code"] = zfill_clean(out["cbsa_code"], 5)
    out["state_fips"] = zfill_clean(out["state_fips"], 2)
    out["county_fips_part"] = zfill_clean(out["county_fips_part"], 3)
    out["county_fips"] = out["state_fips"] + out["county_fips_part"]

    out["cbsa_name"] = out["cbsa_name"].astype(str).str.strip()
    out["county_name"] = out["county_name"].astype(str).str.strip()
    out["state_name"] = out["state_name"].astype(str).str.strip()
    out["metro_micro_type"] = out["metro_micro_type"].astype(str).str.strip()

    if KEEP_METROPOLITAN_ONLY:
        out = out[
            out["metro_micro_type"]
            .str.contains("Metropolitan Statistical Area", case=False, na=False)
        ].copy()

    out = out[[
        "cbsa_code",
        "cbsa_name",
        "metro_micro_type",
        "county_fips",
        "county_name",
        "state_name",
    ]].drop_duplicates()

    out.to_csv(OUT_CROSSWALK, index=False)

    print(f"Saved: {OUT_CROSSWALK}")
    print(out.head())
    print(out.shape)

    return out


# ============================================================
# CENSUS BPS METRO/CBSA BUILDING PERMITS
# ============================================================

BPS_FOLDERS = [
    "https://www2.census.gov/econ/bps/Metro%20%28ending%202023%29/",
    "https://www2.census.gov/econ/bps/CBSA%20%28beginning%20Jan%202024%29/",
]

BPS_COLS = [
    "period", "csa_code", "cbsa_code", "header_code", "cbsa_name",

    "imp_101_bldgs", "imp_101_units", "imp_101_value",
    "imp_103_bldgs", "imp_103_units", "imp_103_value",
    "imp_104_bldgs", "imp_104_units", "imp_104_value",
    "imp_105_bldgs", "imp_105_units", "imp_105_value",

    "rep_101_bldgs", "rep_101_units", "rep_101_value",
    "rep_103_bldgs", "rep_103_units", "rep_103_value",
    "rep_104_bldgs", "rep_104_units", "rep_104_value",
    "rep_105_bldgs", "rep_105_units", "rep_105_value",
]


def list_bps_files():
    files = []

    for folder in BPS_FOLDERS:
        print(f"Reading Census BPS folder: {folder}")

        try:
            html = requests.get(folder, timeout=60).text
        except Exception as e:
            print(f"Could not read folder: {folder}")
            print(e)
            continue

        found = re.findall(r'href="([^"]+\.txt)"', html, flags=re.I)

        for fn in found:
            fn_low = fn.lower()

            if not fn_low.endswith("c.txt"):
                continue

            old_match = re.match(r"ma(\d{2})(\d{2})c\.txt", fn_low)
            new_match = re.match(r"cbsa(\d{2})(\d{2})c\.txt", fn_low)

            if old_match:
                yy = int(old_match.group(1))
                mm = int(old_match.group(2))
            elif new_match:
                yy = int(new_match.group(1))
                mm = int(new_match.group(2))
            else:
                continue

            year = 2000 + yy if yy <= 50 else 1900 + yy

            if year >= START_YEAR and 1 <= mm <= 12:
                files.append({
                    "url": urljoin(folder, fn),
                    "filename": fn,
                    "year": year,
                    "month": mm,
                })

    return sorted(files, key=lambda x: (x["year"], x["month"], x["filename"]))


def parse_one_bps_file(content):
    df = pd.read_csv(
        BytesIO(content),
        header=None,
        names=BPS_COLS,
        dtype=str,
        encoding="latin1",
    )

    df["date"] = pd.to_datetime(
        df["period"].astype(str).str[:6] + "01",
        format="%Y%m%d",
        errors="coerce",
    )

    df["cbsa_code"] = zfill_clean(df["cbsa_code"], 5)
    df["cbsa_name"] = df["cbsa_name"].astype(str).str.strip()
    df["header_code"] = df["header_code"].astype(str).str.strip()

    df = df[df["cbsa_code"] != "99999"].copy()
    df = df[df["date"].notna()].copy()

    if KEEP_METROPOLITAN_ONLY:
        df = df[df["header_code"] != "5"].copy()

    unit_cols = ["imp_101_units", "imp_103_units", "imp_104_units", "imp_105_units"]
    value_cols = ["imp_101_value", "imp_103_value", "imp_104_value", "imp_105_value"]

    for col in unit_cols + value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["permits_units_total"] = df[unit_cols].sum(axis=1)
    df["permits_value_total_thousands"] = df[value_cols].sum(axis=1)

    return df[[
        "cbsa_code",
        "date",
        "cbsa_name",
        "permits_units_total",
        "permits_value_total_thousands",
        "imp_101_units",
        "imp_103_units",
        "imp_104_units",
        "imp_105_units",
    ]].copy()


def download_bps_metro():
    if os.path.exists(OUT_BPS_MONTHLY):
        print(f"Loading existing file: {OUT_BPS_MONTHLY}")
        return pd.read_csv(
            OUT_BPS_MONTHLY,
            parse_dates=["date"],
            dtype={"cbsa_code": str}
        )

    files = list_bps_files()

    if not files:
        raise RuntimeError("No BPS metro/CBSA files found.")

    print(f"Found {len(files)} BPS monthly files.")

    cached_csvs = []

    for i, info in enumerate(files, start=1):
        cache_file = info["filename"].replace(".txt", ".csv")
        cache_path = os.path.join(BPS_CACHE_DIR, cache_file)

        if os.path.exists(cache_path):
            print(f"[{i}/{len(files)}] Already cached: {info['filename']}")
            cached_csvs.append(cache_path)
            continue

        print(f"[{i}/{len(files)}] Downloading BPS: {info['filename']}")

        try:
            content = download_bytes_with_retry(info["url"])
            parsed = parse_one_bps_file(content)
            parsed.to_csv(cache_path, index=False)
            cached_csvs.append(cache_path)
        except Exception as e:
            print(f"  FAILED and skipped: {info['filename']}")
            print(f"  Error: {e}")

    if not cached_csvs:
        raise RuntimeError("No BPS files were successfully downloaded or cached.")

    print("Combining cached BPS files...")

    parts = [
        pd.read_csv(path, parse_dates=["date"], dtype={"cbsa_code": str})
        for path in cached_csvs
    ]

    bps = pd.concat(parts, ignore_index=True)
    bps = bps.dropna(subset=["cbsa_code", "date"])
    bps = bps.sort_values(["cbsa_code", "date"])

    numeric_cols = [
        "permits_units_total",
        "permits_value_total_thousands",
        "imp_101_units",
        "imp_103_units",
        "imp_104_units",
        "imp_105_units",
    ]

    name_lookup = (
        bps[["cbsa_code", "cbsa_name"]]
        .dropna()
        .drop_duplicates("cbsa_code")
    )

    bps = (
        bps.groupby(["cbsa_code", "date"], as_index=False)[numeric_cols]
        .sum()
        .merge(name_lookup, on="cbsa_code", how="left")
    )

    bps = bps.sort_values(["cbsa_code", "date"])

    bps["permits_units_growth_12m"] = (
        bps.groupby("cbsa_code")["permits_units_total"].pct_change(12)
    )

    bps.to_csv(OUT_BPS_MONTHLY, index=False)

    print(f"Saved: {OUT_BPS_MONTHLY}")
    print(bps.head())
    print(bps.shape)

    return bps


# ============================================================
# BEA COUNTY DATA
# ============================================================

def require_bea_key():
    if not BEA_KEY:
        raise ValueError(
            "BEA_KEY is missing.\n"
            "In Git Bash run:\n"
            '    export BEA_API_KEY="YOUR_REAL_BEA_KEY"\n'
            "Then run:\n"
            "    python zillow-covariate.py\n"
            "Or paste the key into BEA_KEY near the top of this script."
        )


def bea_get_regional_county(table_name, line_code, value_name):
    """
    Download a BEA Regional table for all counties.
    """
    require_bea_key()

    params = {
        "UserID": BEA_KEY,
        "method": "GetData",
        "datasetname": "Regional",
        "TableName": table_name,
        "LineCode": str(line_code),
        "GeoFips": "COUNTY",
        "Year": "ALL",
        "ResultFormat": "JSON",
    }

    print(f"Calling BEA API: TableName={table_name}, LineCode={line_code}, GeoFips=COUNTY")

    response = requests.get(
        "https://apps.bea.gov/api/data",
        params=params,
        timeout=180,
    )
    response.raise_for_status()

    js = response.json()
    results = js.get("BEAAPI", {}).get("Results", {})

    if "Data" not in results:
        print("BEA response did not contain Data. First part of response:")
        print(json.dumps(js, indent=2)[:5000])
        raise ValueError(f"BEA returned no Data for {table_name}, LineCode={line_code}")

    df = pd.DataFrame(results["Data"])

    geo_col = find_col_case_insensitive(df, "GeoFips")
    name_col = find_col_case_insensitive(df, "GeoName")
    time_col = find_col_case_insensitive(df, "TimePeriod")
    value_col = find_col_case_insensitive(df, "DataValue")

    df["county_fips"] = zfill_clean(df[geo_col], 5)
    df["year"] = pd.to_numeric(df[time_col], errors="coerce")

    df[value_name] = (
        df[value_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(NA)", "", regex=False)
        .str.strip()
    )
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    df = df[
        df["year"].notna()
        & df["county_fips"].str.len().eq(5)
        & ~df["county_fips"].str.endswith("000")
    ].copy()

    df["year"] = df["year"].astype(int)

    out = df[["county_fips", name_col, "year", value_name]].rename(
        columns={name_col: "county_name_bea"}
    )

    return out.sort_values(["county_fips", "year"])


def download_county_gdp():
    if os.path.exists(OUT_COUNTY_GDP_ANNUAL):
        print(f"Loading existing file: {OUT_COUNTY_GDP_ANNUAL}")
        return pd.read_csv(
            OUT_COUNTY_GDP_ANNUAL,
            dtype={"county_fips": str}
        )

    gdp = bea_get_regional_county(
        table_name="CAGDP9",
        line_code=1,
        value_name="real_gdp"
    )

    gdp = gdp[gdp["year"] >= 2001].copy()

    gdp.to_csv(OUT_COUNTY_GDP_ANNUAL, index=False)

    print(f"Saved: {OUT_COUNTY_GDP_ANNUAL}")
    print(gdp.head())
    print(gdp.shape)

    return gdp


def aggregate_county_gdp_to_msa(county_gdp, crosswalk):
    if os.path.exists(OUT_MSA_GDP_ANNUAL):
        print(f"Loading existing file: {OUT_MSA_GDP_ANNUAL}")
        return pd.read_csv(
            OUT_MSA_GDP_ANNUAL,
            dtype={"cbsa_code": str}
        )

    temp = county_gdp.merge(
        crosswalk[["county_fips", "cbsa_code", "cbsa_name"]],
        on="county_fips",
        how="inner"
    )

    msa = (
        temp.groupby(["cbsa_code", "cbsa_name", "year"], as_index=False)["real_gdp"]
        .sum()
        .sort_values(["cbsa_code", "year"])
    )

    msa["real_gdp_growth_1y"] = (
        msa.groupby("cbsa_code")["real_gdp"].pct_change(1)
    )

    msa.to_csv(OUT_MSA_GDP_ANNUAL, index=False)

    print(f"Saved: {OUT_MSA_GDP_ANNUAL}")
    print(msa.head())
    print(msa.shape)

    return msa


def annual_gdp_to_monthly_lagged(msa_gdp):
    if os.path.exists(OUT_MSA_GDP_MONTHLY):
        print(f"Loading existing file: {OUT_MSA_GDP_MONTHLY}")
        return pd.read_csv(
            OUT_MSA_GDP_MONTHLY,
            parse_dates=["date"],
            dtype={"cbsa_code": str}
        )

    months = pd.DataFrame({"month": range(1, 13)})

    gm = (
        msa_gdp.assign(year_for_month=msa_gdp["year"] + 1)
        [["cbsa_code", "year_for_month", "real_gdp", "real_gdp_growth_1y"]]
        .merge(months, how="cross")
    )

    gm["date"] = pd.to_datetime(dict(
        year=gm["year_for_month"],
        month=gm["month"],
        day=1,
    ))

    gm = gm[[
        "cbsa_code",
        "date",
        "real_gdp",
        "real_gdp_growth_1y",
    ]].copy()

    gm = gm.sort_values(["cbsa_code", "date"])
    gm.to_csv(OUT_MSA_GDP_MONTHLY, index=False)

    print(f"Saved: {OUT_MSA_GDP_MONTHLY}")
    print(gm.head())
    print(gm.shape)

    return gm


def download_county_population():
    if os.path.exists(OUT_COUNTY_POP_ANNUAL):
        print(f"Loading existing file: {OUT_COUNTY_POP_ANNUAL}")
        return pd.read_csv(
            OUT_COUNTY_POP_ANNUAL,
            dtype={"county_fips": str}
        )

    pop = bea_get_regional_county(
        table_name="CAINC1",
        line_code=2,
        value_name="population"
    )

    pop = pop[pop["year"] >= START_YEAR].copy()
    pop.to_csv(OUT_COUNTY_POP_ANNUAL, index=False)

    print(f"Saved: {OUT_COUNTY_POP_ANNUAL}")
    print(pop.head())
    print(pop.shape)

    return pop


def aggregate_county_population_to_msa(county_pop, crosswalk):
    if os.path.exists(OUT_MSA_POP_ANNUAL):
        print(f"Loading existing file: {OUT_MSA_POP_ANNUAL}")
        return pd.read_csv(
            OUT_MSA_POP_ANNUAL,
            dtype={"cbsa_code": str}
        )

    temp = county_pop.merge(
        crosswalk[["county_fips", "cbsa_code", "cbsa_name"]],
        on="county_fips",
        how="inner"
    )

    msa = (
        temp.groupby(["cbsa_code", "cbsa_name", "year"], as_index=False)["population"]
        .sum()
        .sort_values(["cbsa_code", "year"])
    )

    msa["population_growth_1y"] = (
        msa.groupby("cbsa_code")["population"].pct_change(1)
    )

    msa.to_csv(OUT_MSA_POP_ANNUAL, index=False)

    print(f"Saved: {OUT_MSA_POP_ANNUAL}")
    print(msa.head())
    print(msa.shape)

    return msa


def annual_population_to_monthly_interpolated(msa_pop):
    if os.path.exists(OUT_MSA_POP_MONTHLY):
        print(f"Loading existing file: {OUT_MSA_POP_MONTHLY}")
        return pd.read_csv(
            OUT_MSA_POP_MONTHLY,
            parse_dates=["date"],
            dtype={"cbsa_code": str}
        )

    msa_pop = msa_pop.copy()

    # Treat annual population as July 1.
    msa_pop["date"] = pd.to_datetime(msa_pop["year"].astype(str) + "-07-01")

    min_date = pd.Timestamp(f"{START_YEAR}-01-01")
    max_year = int(msa_pop["year"].max())
    max_date = pd.Timestamp(f"{max_year}-12-01")
    month_index = pd.date_range(min_date, max_date, freq="MS")

    def one_msa(g):
        s = (
            g.drop_duplicates("date")
            .sort_values("date")
            .set_index("date")["population"]
        )

        s_monthly = (
            s.reindex(s.index.union(month_index))
            .sort_index()
            .interpolate("time")
            .reindex(month_index)
        )

        return pd.DataFrame({
            "cbsa_code": g.name,
            "date": month_index,
            "population": s_monthly.to_numpy(),
        })

    pm = (
        msa_pop.groupby("cbsa_code", group_keys=False)
        .apply(one_msa)
        .reset_index(drop=True)
    )

    pm = pm.sort_values(["cbsa_code", "date"])

    pm["population_growth_12m"] = (
        pm.groupby("cbsa_code")["population"].pct_change(12)
    )

    pm.to_csv(OUT_MSA_POP_MONTHLY, index=False)

    print(f"Saved: {OUT_MSA_POP_MONTHLY}")
    print(pm.head())
    print(pm.shape)

    return pm


# ============================================================
# FINAL MERGE
# ============================================================

def merge_all(bps, gdp_monthly, pop_monthly):
    bps["cbsa_code"] = bps["cbsa_code"].astype(str).str.zfill(5)
    gdp_monthly["cbsa_code"] = gdp_monthly["cbsa_code"].astype(str).str.zfill(5)
    pop_monthly["cbsa_code"] = pop_monthly["cbsa_code"].astype(str).str.zfill(5)

    df = (
        bps.merge(pop_monthly, on=["cbsa_code", "date"], how="left")
        .merge(gdp_monthly, on=["cbsa_code", "date"], how="left")
    )

    df["permits_per_1000_people"] = (
        1000 * df["permits_units_total"] / df["population"]
    )

    df = df.sort_values(["cbsa_code", "date"])
    df.to_csv(OUT_FINAL, index=False)

    print(f"Saved: {OUT_FINAL}")
    print(df.head())
    print(df.shape)

    missing_pop = df["population"].isna().mean()
    missing_gdp = df["real_gdp"].isna().mean()

    print(f"Missing population share: {missing_pop:.2%}")
    print(f"Missing GDP share: {missing_gdp:.2%}")

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Script started.")
    print(f"START_YEAR = {START_YEAR}")
    print(f"KEEP_METROPOLITAN_ONLY = {KEEP_METROPOLITAN_ONLY}")
    print()

    print("Step 1/7: Census CBSA-county crosswalk")
    crosswalk = download_cbsa_county_crosswalk()
    print()

    print("Step 2/7: Census BPS metro/CBSA monthly permits")
    bps = download_bps_metro()
    print()

    print("Step 3/7: BEA county real GDP, then aggregate to MSA")
    county_gdp = download_county_gdp()
    msa_gdp = aggregate_county_gdp_to_msa(county_gdp, crosswalk)
    gdp_monthly = annual_gdp_to_monthly_lagged(msa_gdp)
    print()

    print("Step 4/7: BEA county population, then aggregate to MSA")
    county_pop = download_county_population()
    msa_pop = aggregate_county_population_to_msa(county_pop, crosswalk)
    pop_monthly = annual_population_to_monthly_interpolated(msa_pop)
    print()

    print("Step 5/7: Merge all monthly covariates")
    final = merge_all(bps, gdp_monthly, pop_monthly)
    print()

    print("Done.")
    print(f"Final file: {OUT_FINAL}")
