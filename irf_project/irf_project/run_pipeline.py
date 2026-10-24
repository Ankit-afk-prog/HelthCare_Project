"""Run the whole project end-to-end (checkpointed):  python run_pipeline.py [stage ...]
Stages: data  models  modelsB  design2  shap  summary   (default: all)"""
import sys, json, time, platform, joblib
import numpy as np, pandas as pd, sklearn, xgboost, shap, scipy
from src import pipeline as P, design2 as D2, evaluation as ev
from src.config import *

CK = PROC / "cache"; CK.mkdir(exist_ok=True)
stages = sys.argv[1:] or ["data", "models", "modelsB", "design2", "shap", "summary"]
t0 = time.time(); R = {}
def ck(n, obj=None):
    f = CK / f"{n}.joblib"
    if obj is not None: joblib.dump(obj, f); return obj
    return joblib.load(f)

if "data" in stages:
    merged, info_data = P.stage_data()
    df, Pm, sens, wtab, contrib, val, info_idx = P.stage_index(merged)
    P.stage_eda(merged, df, Pm); tests, pw = P.stage_stats(merged, df)
    ck("data", dict(info_data=info_data, info_idx=info_idx, val=val)); print("data stage done", info_idx, flush=True)
merged = pd.read_csv(PROC / "irf_facilities_merged.csv", dtype={CCN: str}, parse_dates=["Certification Date"])
df = pd.read_csv(PROC / "facility_quality_index.csv", dtype={CCN: str}); df["performance_category"] = pd.Categorical(df["performance_category"], CLASS_ORDER)

if "models" in stages:
    X, XB, y = P._feature_sets(merged, df)
    A = P.run_experiment(X, y, "A_characteristics"); ck("A", A)
    ev.plot_confusion(A["probas"], A["yte"], FIG / "fig11_confusion_matrices.png")
    ev.plot_roc_pr(A["probas"], A["yte"], FIG / "fig12_roc_curves.png", FIG / "fig13_pr_curves.png")
    ev.plot_comparison(A["res"], FIG / "fig14_model_comparison.png"); ev.plot_calibration(A["probas"], A["yte"], FIG / "fig15_calibration_high.png")
    dist = y.map(dict(enumerate(CLASS_ORDER))).value_counts().reindex(CLASS_ORDER).astype(int).to_dict()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from src import feature_engineering as fe, model_training as mt
    Pm = fe.direction_adjusted_percentiles(merged[list(INDEX_MEASURES)])[df["quality_index"].notna().values]
    leak = cross_val_score(Pipeline([("i", SimpleImputer()), ("rf", RandomForestClassifier(200, random_state=SEED))]), Pm, y, cv=mt.CV, scoring="accuracy").mean()
    ck("misc", dict(dist=dist, leak=float(leak))); print("models A done", flush=True)

if "modelsB" in stages:
    X, XB, y = P._feature_sets(merged, df)
    B = P.run_experiment(XB, y, "B_plus_heldout_quality", save=False); ck("B", B)
    ev.plot_comparison(B["res"], FIG / "fig16_model_comparison_setB.png"); print("models B done", flush=True)

if "design2" in stages:
    C, info2, _ = D2.stage_design2(merged, df); ck("C", C); ck("info2", info2)
    ev.plot_comparison(C["res"], FIG / "fig22_model_comparison_design2.png"); print("design2 done", info2, flush=True)

if "shap" in stages:
    from src import explainability as ex
    A = ck("A"); imp, local = P.stage_shap(A, df)
    rf = A["models"]["Random Forest"]; names = ex.feature_names(rf)
    import matplotlib.pyplot as plt
    s = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5)); s.plot.barh(ax=ax, color="#4c72b0"); ax.set(xlabel="Impurity-based importance", title="Random Forest feature importance (design 1, set A)")
    fig.tight_layout(); fig.savefig(FIG / "fig17_rf_importance.png", dpi=140); plt.close(fig)
    # SHAP for design 2 (XGBoost)
    C = ck("C"); svC, XtC, nmC = ex.shap_values(C["models"]["XGBoost"], C["Xte"]); impC = ex.global_importance(svC, nmC); impC.to_csv(TAB / "shap_importance_design2.csv")
    ex.plot_importance(impC, FIG / "fig23_shap_importance_design2.png"); ex.plot_beeswarm(svC, XtC, nmC, 2, FIG / "fig24_shap_beeswarm_design2_high.png")
    print("shap done", flush=True)

if "summary" in stages:
    d = ck("data"); m = ck("misc"); A = ck("A")
    res = {"data": d["info_data"], "index": d["info_idx"], "class_distribution_scored": m["dist"], "leakage_demo_cv_accuracy_index_measures_as_features": m["leak"],
           "voting_weights": {k: float(v) for k, v in A["vw"].items()}, "n_train": int(len(A["Xtr"])), "n_test": int(len(A["Xte"])), "design2": ck("info2"),
           "env": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "shap": shap.__version__},
           "seed": SEED}
    json.dump(res, open(OUT / "results.json", "w"), indent=1, default=str); print("summary written")
print("DONE", round(time.time() - t0), "s", flush=True)
