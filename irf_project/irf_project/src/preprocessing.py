"""Loading, cleaning, pivoting and safe merging of the two CMS IRF files."""
import numpy as np
import pandas as pd
from .config import *


def load_raw():
    gen = pd.read_csv(GENERAL_FILE, dtype=str)
    prov = pd.read_csv(PROVIDER_FILE, dtype=str)
    return gen, prov


def inspect_table(df, name):
    """File-level summary used in the data-understanding notebook."""
    s = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing_%": (df.isna().mean() * 100).round(2),
                      "n_unique": df.nunique(), "example": df.iloc[0]})
    s.insert(0, "file", name)
    return s


def clean_provider(prov):
    """Provider file is long (facility x measure). Score is text: numbers may contain
    commas ('1,050') or be CMS labels ('Not Available', 'Better than the National Rate')."""
    p = prov.copy()
    p["score_num"] = pd.to_numeric(p["Score"].str.replace(",", "", regex=False), errors="coerce")
    p["score_label"] = np.where(p["score_num"].isna(), p["Score"], np.nan)
    p["Footnote"] = p["Footnote"].replace("-", np.nan)
    return p


def pivot_provider(p):
    """One row per CCN, one numeric column per measure code (checks 1:1 after pivot)."""
    assert not p.duplicated([CCN, "Measure Code"]).any(), "duplicate CCN x measure"
    wide = p.pivot(index=CCN, columns="Measure Code", values="score_num")
    wide.columns.name = None
    return wide


def comp_labels(p):
    """CMS comparative labels (kept only for external validation of the index)."""
    c = p[p["Measure Code"].isin(COMP_CODES)].pivot(index=CCN, columns="Measure Code", values="Score")
    c.columns.name = None
    return c.replace("Not Available", np.nan)


def clean_general(gen):
    g = gen.copy()
    g["Certification Date"] = pd.to_datetime(g["Certification Date"], format="%m/%d/%Y", errors="coerce")
    g["Ownership Type"] = g["Ownership Type"].str.strip()
    g["CMS Region"] = g["CMS Region"].astype(int)
    g["Address Line 2"] = g["Address Line 2"].replace("-", np.nan)
    return g


def merge_facilities(gen, wide):
    """Left-join the pivoted (1 row/CCN) provider data onto General Information (1 row/CCN)."""
    n0 = len(gen)
    assert gen[CCN].is_unique and wide.index.is_unique
    m = gen.merge(wide.reset_index(), on=CCN, how="left", validate="one_to_one")
    assert len(m) == n0, "merge changed number of facilities"
    return m, {"general_rows": n0, "provider_long_rows": None, "pivoted_rows": len(wide), "merged_rows": len(m)}


def invalid_value_report(wide):
    """Range checks on index measures (rates 0-100, SIR/ratio >= 0)."""
    rows = []
    for c, (nm, d, dom) in INDEX_MEASURES.items():
        s = wide[c].dropna()
        pct = c.endswith(("OBS_RATE", "ADJ_RATE", "RS_RATE"))
        bad = int(((s < 0) | (s > 100)).sum()) if pct else int((s < 0).sum())
        rows.append({"measure": c, "name": nm, "n_valid": len(s), "min": s.min(), "max": s.max(), "invalid_values": bad})
    return pd.DataFrame(rows)


def outlier_report(df, cols):
    """IQR-rule outlier counts. Outliers are REPORTED, not deleted (they can be genuine facilities)."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"measure": c, "n": len(s), "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": int(((s < lo) | (s > hi)).sum())})
    return pd.DataFrame(rows)
