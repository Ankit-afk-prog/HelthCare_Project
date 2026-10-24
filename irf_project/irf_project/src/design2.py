"""Design 2 (secondary experiment): OUTCOME-based index vs. CARE-PROCESS + characteristic features.

Index (target)  : 8 outcome/safety/cost measures (domains Safety, Readmission & community return, Function, Cost).
Features        : facility characteristics + the 3 care-process measures (I_021, I_024, I_025) + 2 held-out measures
                  (I_016 staff flu vaccination, I_006 CAUTI SIR). No feature is part of the index -> no leakage.
Research question: do care-process indicators and facility characteristics carry information about outcome-based performance?
"""
import numpy as np, pandas as pd
from . import feature_engineering as fe, pipeline as P
from .config import *

OUTCOME = [c for c, v in INDEX_MEASURES.items() if v[2] != "Care processes"]
PROCESS = [c for c, v in INDEX_MEASURES.items() if v[2] == "Care processes"]
MIN_OUT = 6


def stage_design2(m, df):
    Pm = fe.direction_adjusted_percentiles(m[OUTCOME])
    idx = fe.weighted_index(Pm, pd.Series(1.0, index=OUTCOME), min_measures=MIN_OUT)
    cat, cuts = fe.to_categories(idx); ok = idx.notna()
    X = fe.engineer_features(m)[ok]
    for c in PROCESS + HELD_OUT_MEASURES: X[c] = m.loc[ok, c]
    y = cat[ok].map({c: i for i, c in enumerate(CLASS_ORDER)}).astype(int)
    out = m[[CCN, "Provider Name"]].copy(); out["outcome_index"] = idx; out["outcome_category"] = cat
    out.to_csv(PROC / "facility_outcome_index_design2.csv", index=False)
    C = P.run_experiment(X, y, "C_outcome_index_process_features")
    info = {"n_scored": int(ok.sum()), "tercile_cuts": [float(c) for c in cuts], "class_counts": cat.value_counts().reindex(CLASS_ORDER).astype(int).to_dict(), "min_measures": MIN_OUT}
    return C, info, out
