"""Quality Performance Index (target construction) and leakage-safe feature engineering."""
import numpy as np
import pandas as pd
from .config import *


def direction_adjusted_percentiles(df):
    """Percentile rank in [0,1]; direction-flipped so that 1 = best for every measure."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        d = INDEX_MEASURES[c][1]
        r = df[c].rank(pct=True, method="average")
        out[c] = (1 - r + 1 / df[c].notna().sum()) if d == "lower" else r
    return out.clip(0, 1)


def weights_equal():
    return pd.Series(1.0, index=list(INDEX_MEASURES))


def weights_domain():
    """Each domain gets the same total weight; measures share it equally inside the domain."""
    doms = pd.Series({c: v[2] for c, v in INDEX_MEASURES.items()})
    return 1.0 / doms.map(doms.value_counts()) / doms.nunique()


def weights_pca(P):
    """Absolute first-principal-component loadings (data-driven alternative; sign-agnostic)."""
    X = P.fillna(P.mean())
    Z = (X - X.mean()) / X.std()
    _, _, vt = np.linalg.svd(Z.values, full_matrices=False)
    return pd.Series(np.abs(vt[0]), index=P.columns)


def weighted_index(P, w, min_measures=MIN_MEASURES):
    """Weighted mean over AVAILABLE measures (weights renormalised per facility).
    Facilities with < min_measures available values get NaN (excluded, never imputed)."""
    avail = P.notna()
    num = (P.fillna(0) * w).sum(axis=1)
    den = (avail * w).sum(axis=1)
    idx = num / den
    idx[avail.sum(axis=1) < min_measures] = np.nan
    return idx * 100


def to_categories(index):
    """Tertile cut-points -> project-specific analytical performance categories."""
    q = index.quantile([1 / 3, 2 / 3]).values
    cat = pd.cut(index, [-np.inf, q[0], q[1], np.inf], labels=CLASS_ORDER)
    return cat, q


def engineer_features(df, n_in_state=None):
    """Facility-characteristic features. None is computed from any index measure."""
    f = pd.DataFrame(index=df.index)
    f["ownership"] = df["Ownership Type"]
    f["cms_region"] = df["CMS Region"].astype(str)
    f["facility_age_years"] = REFERENCE_YEAR - df["Certification Date"].dt.year          # years since Medicare certification
    f["n_irf_in_state"] = df.groupby("State")[CCN].transform("count")                   # local IRF density
    name = df["Provider Name"].str.upper()
    f["name_has_rehab"] = name.str.contains("REHAB", regex=False).astype(int)            # heuristic: free-standing rehab name
    f["name_encompass_chain"] = name.str.contains("ENCOMPASS", regex=False).astype(int)  # largest national IRF operator (keyword flag)
    f["log_eligible_stays"] = np.log1p(df["I_017_01_PPR_PD_VOLUME"])                    # volume (size), not a quality value
    f["log_mspb_episodes"] = np.log1p(df["I_020_01_MSPB_NUMB"])
    f["log_function_cases"] = np.log1p(df["I_026_01_DENOMINATOR"])
    return f


FEATURE_DICTIONARY = pd.DataFrame([
    ("ownership", "Ownership Type (General file)", "categorical", "Ownership differs in incentives/structure; descriptive characteristic."),
    ("cms_region", "CMS Region 1-10", "categorical", "Geography/administrative region."),
    ("facility_age_years", "2026 - year(Certification Date)", "numeric", "Years in the Medicare programme; experience proxy."),
    ("n_irf_in_state", "count of IRFs in the facility's state", "numeric", "Local IRF market density (computed from General file only)."),
    ("name_has_rehab", "1 if 'REHAB' in Provider Name", "binary", "Heuristic for free-standing rehabilitation hospital vs unit/other."),
    ("name_encompass_chain", "1 if 'ENCOMPASS' in Provider Name", "binary", "Heuristic flag for the largest national IRF operator."),
    ("log_eligible_stays", "ln(1 + I_017_01_PPR_PD_VOLUME)", "numeric", "Facility size/volume (eligible stays); log reduces skew."),
    ("log_mspb_episodes", "ln(1 + I_020_01_MSPB_NUMB)", "numeric", "Volume of eligible Medicare episodes."),
    ("log_function_cases", "ln(1 + I_026_01_DENOMINATOR)", "numeric", "Volume of assessed patients."),
    ("I_016_01_OBS_RATE (set B only)", "HCP flu-vaccination rate", "numeric", "Held-out quality measure, NOT in the index (ablation set B)."),
    ("I_006_01_SIR (set B only)", "CAUTI standardized infection ratio", "numeric", "Held-out quality measure, NOT in the index (ablation set B)."),
], columns=["feature", "formula / source", "type", "purpose / rationale"])
