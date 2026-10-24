"""Stage functions used by both run_pipeline.py and the notebooks. Every number in the report comes from here."""
import json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, accuracy_score
from . import preprocessing as pp, feature_engineering as fe, model_training as mt, evaluation as ev, explainability as ex
from .config import *

sns.set_theme(style="whitegrid", context="notebook")
PAL = {"Low": "#d95f02", "Medium": "#7570b3", "High": "#1b9e77"}


def _save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight", dpi=140); plt.close(fig)


# ------------------------------------------------------------------ 1. data
def stage_data():
    gen, prov = pp.load_raw()
    insp = pd.concat([pp.inspect_table(gen, "General Information"), pp.inspect_table(prov, "Provider Data")])
    insp.to_csv(TAB / "file_inspection.csv")
    p = pp.clean_provider(prov); wide = pp.pivot_provider(p); labels = pp.comp_labels(p)
    g = pp.clean_general(gen)
    merged, info = pp.merge_facilities(g, wide); info["provider_long_rows"] = len(prov)
    merged = merged.merge(labels.reset_index(), on=CCN, how="left", validate="one_to_one")
    assert len(merged) == info["general_rows"]
    merged.to_csv(PROC / "irf_facilities_merged.csv", index=False)
    inv = pp.invalid_value_report(wide); inv.to_csv(TAB / "invalid_value_report.csv", index=False)
    out = pp.outlier_report(merged, list(INDEX_MEASURES)); out.to_csv(TAB / "outlier_report.csv", index=False)
    foot = p.groupby(["Footnote"], dropna=False).size().rename("rows").reset_index(); foot.to_csv(TAB / "footnote_counts.csv", index=False)
    info.update({"provider_measures_per_facility": int(prov.groupby(CCN).size().iloc[0]),
                 "n_measure_codes": int(prov["Measure Code"].nunique()),
                 "text_labels_rows": int(p["score_num"].isna().sum()),
                 "general_dups": int(gen.duplicated().sum()), "provider_dups": int(prov.duplicated().sum()),
                 "all_ccn_matched": bool(prov[CCN].isin(gen[CCN]).all() and gen[CCN].isin(prov[CCN]).all()),
                 "invalid_values_total": int(inv["invalid_values"].sum())})
    return merged, info


# ------------------------------------------------------------------ 2. index
def stage_index(m):
    P = fe.direction_adjusted_percentiles(m[list(INDEX_MEASURES)])
    W = {"Equal weights": fe.weights_equal(), "Domain-balanced": fe.weights_domain(), "PCA (PC1) loadings": fe.weights_pca(P)}
    idx = {k: fe.weighted_index(P, w) for k, w in W.items()}
    main = idx["Equal weights"]; cat, cuts = fe.to_categories(main)
    df = m[[CCN, "Provider Name", "State", "CMS Region", "Ownership Type"]].copy()
    df["n_measures_available"] = P.notna().sum(1)
    for k, v in idx.items(): df["index_" + k.split()[0].lower()] = v
    df["quality_index"] = main; df["performance_category"] = cat
    # sensitivity
    cats = {k: fe.to_categories(v)[0] for k, v in idx.items()}
    ok = main.notna()
    sens = []
    for k in list(W)[1:]:
        both = ok & idx[k].notna()
        sens.append({"alternative": k, "spearman_vs_equal": stats.spearmanr(main[both], idx[k][both])[0],
                     "category_agreement_%": 100 * (cats["Equal weights"][both] == cats[k][both]).mean(),
                     "cohen_kappa": cohen_kappa_score(cats["Equal weights"][both], cats[k][both])})
    sens = pd.DataFrame(sens)
    wtab = pd.DataFrame(W).round(4); wtab.insert(0, "measure", [SHORT[c] for c in wtab.index]); wtab.insert(1, "better", [INDEX_MEASURES[c][1] for c in wtab.index])
    wtab.insert(2, "domain", [INDEX_MEASURES[c][2] for c in wtab.index])
    # contribution of each measure (corr of its percentile with the index)
    contrib = pd.DataFrame({"measure": [SHORT[c] for c in P.columns],
                            "n_available": P.notna().sum().values,
                            "spearman_with_index": [stats.spearmanr(P[c][ok], main[ok], nan_policy="omit")[0] for c in P.columns]})
    # external validation vs CMS per-measure comparative labels (partly circular -> descriptive only)
    val = None
    if "I_017_01_PPR_PD_COMP_PERF" in m:
        t = pd.concat([cat, m["I_019_02_DTC_COMP_PERF"].str.replace(" than the National Rate", "")], axis=1).dropna()
        t.columns = ["category", "CMS label (discharge to community)"]
        val = pd.crosstab(t["category"], t["CMS label (discharge to community)"], normalize="index").mul(100).round(1)
    df.to_csv(PROC / "facility_quality_index.csv", index=False)
    sens.to_csv(TAB / "index_sensitivity.csv", index=False); wtab.to_csv(TAB / "index_weights.csv", index=False)
    contrib.to_csv(TAB / "index_measure_contribution.csv", index=False)
    info = {"n_scored": int(ok.sum()), "n_excluded_lt_min": int((~ok).sum()), "min_measures": MIN_MEASURES,
            "tercile_cuts": [float(c) for c in cuts], "class_counts": cat.value_counts().reindex(CLASS_ORDER).astype(int).to_dict(),
            "index_mean": float(main.mean()), "index_sd": float(main.std()), "index_min": float(main.min()), "index_max": float(main.max()),
            "measures_available_dist": df["n_measures_available"].value_counts().sort_index().to_dict()}
    return df, P, sens, wtab, contrib, val, info


# ------------------------------------------------------------------ 3. EDA
def stage_eda(m, df, P):
    d = df[df["quality_index"].notna()].copy()
    d["log_stays"] = np.log1p(m.loc[d.index, "I_017_01_PPR_PD_VOLUME"])
    # 1 missingness
    cols = list(INDEX_MEASURES) + HELD_OUT_MEASURES + VOLUME_CODES
    miss = (m[cols].isna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6)); miss.rename(lambda c: SHORT.get(c, c)).plot.barh(ax=ax, color="#4c72b0")
    ax.set(xlabel="Missing facilities (%)", title="Missing values in selected quality and volume variables (n=1,222 facilities)"); _save(fig, "fig01_missingness.png")
    # 2 category distribution
    fig, ax = plt.subplots(figsize=(5.5, 4)); c = d["performance_category"].value_counts().reindex(CLASS_ORDER)
    ax.bar(c.index, c.values, color=[PAL[k] for k in c.index]); [ax.text(i, v + 3, int(v), ha="center") for i, v in enumerate(c.values)]
    ax.set(ylabel="Number of facilities", xlabel="Project-specific performance category", title="Distribution of performance categories"); _save(fig, "fig02_category_distribution.png")
    # 3 state / region
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    s = m["State"].value_counts().head(15)[::-1]; axs[0].barh(s.index, s.values, color="#4c72b0"); axs[0].set(xlabel="Facilities", title="Top 15 states by number of IRFs")
    r = m["CMS Region"].value_counts().sort_index(); axs[1].bar(r.index.astype(str), r.values, color="#55a868"); axs[1].set(xlabel="CMS Region", ylabel="Facilities", title="IRFs by CMS Region"); _save(fig, "fig03_geography.png")
    # 4 ownership
    fig, ax = plt.subplots(figsize=(5.5, 4)); o = m["Ownership Type"].value_counts()
    ax.bar(o.index, o.values, color="#8172b2"); [ax.text(i, v + 5, int(v), ha="center") for i, v in enumerate(o.values)]
    ax.set(ylabel="Facilities", title="Facilities by ownership type"); _save(fig, "fig04_ownership.png")
    # 5 distributions
    fig, axs = plt.subplots(3, 4, figsize=(15, 9)); axs = axs.ravel()
    for a, c in zip(axs, INDEX_MEASURES):
        sns.histplot(m[c].dropna(), bins=30, ax=a, color="#4c72b0"); a.set_title(SHORT[c], fontsize=8); a.set_xlabel(""); a.set_ylabel("Facilities")
    axs[-1].axis("off"); fig.suptitle("Distributions of the 11 index quality measures (raw scale)", y=1.0); _save(fig, "fig05_measure_distributions.png")
    # index hist
    fig, ax = plt.subplots(figsize=(7, 4)); sns.histplot(d["quality_index"], bins=30, ax=ax, color="#4c72b0")
    for q in fe.to_categories(df["quality_index"])[1]: ax.axvline(q, color="k", ls="--")
    ax.set(xlabel="Quality Performance Index (0-100)", ylabel="Facilities", title="Quality Performance Index with tercile cut-points"); _save(fig, "fig06_index_distribution.png")
    # 6 correlation
    corr = m[list(INDEX_MEASURES)].corr(method="spearman").rename(index=lambda c: SHORT[c][:34], columns=lambda c: SHORT[c][:34])
    fig, ax = plt.subplots(figsize=(10, 8)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7}, cbar_kws={"label": "Spearman rho"})
    ax.set_title("Spearman correlation between raw quality measures (pairwise complete)"); _save(fig, "fig07_correlation_heatmap.png")
    # 7 boxplots
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(data=d, x="Ownership Type", y="quality_index", ax=axs[0], color="#a6cee3", order=["For profit", "Government", "Non-profit", "Physician"]); axs[0].set(ylabel="Quality Performance Index", title="Index by ownership")
    sns.boxplot(data=d, x="CMS Region", y="quality_index", ax=axs[1], color="#b2df8a"); axs[1].set(ylabel="Quality Performance Index", title="Index by CMS Region"); _save(fig, "fig08_boxplots_ownership_region.png")
    # 8 index vs volume
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); sns.scatterplot(data=d, x="log_stays", y="quality_index", hue="performance_category", palette=PAL, hue_order=CLASS_ORDER, ax=ax, s=18)
    ax.set(xlabel="ln(1 + eligible stays)", ylabel="Quality Performance Index", title="Index vs facility volume"); _save(fig, "fig09_index_vs_volume.png")
    # 9 raw measures by category (boxplots, 6 chosen)
    sel = ["I_017_01_PPR_PD_RSRR", "I_019_02_DTC_RS_RATE", "I_026_01_OBS_RATE", "I_022_01_ADJ_RATE", "I_021_01_OBS_RATE", "I_020_01_MSPB_SCORE"]
    t = m.loc[d.index, sel].join(d["performance_category"]); fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    for a, c in zip(axs.ravel(), sel):
        sns.boxplot(data=t, x="performance_category", y=c, order=CLASS_ORDER, palette=PAL, ax=a); a.set(title=SHORT[c][:42], xlabel="", ylabel="")
    fig.suptitle("Selected raw measures by performance category", y=1.0); _save(fig, "fig10_measures_by_category.png")
    # tables
    desc = m[list(INDEX_MEASURES)].describe().T.rename(index=SHORT).round(3); desc.to_csv(TAB / "descriptive_measures.csv")
    own = pd.concat([m["Ownership Type"].value_counts().rename("facilities"), d.groupby("Ownership Type")["quality_index"].agg(["median", "mean"]).round(2)], axis=1)
    reg = pd.concat([m["CMS Region"].value_counts().sort_index().rename("facilities"), d.groupby("CMS Region")["quality_index"].agg(["median", "mean"]).round(2)], axis=1)
    own.to_csv(TAB / "ownership_summary.csv"); reg.to_csv(TAB / "region_summary.csv")
    return desc, own, reg, miss


# ------------------------------------------------------------------ 4. diagnostic statistics
def stage_stats(m, df):
    d = df[df["quality_index"].notna()].copy(); rows = []
    def kw(col):
        gs = [g["quality_index"].values for _, g in d.groupby(col) if len(g) >= 5]
        h, p = stats.kruskal(*gs); n = sum(len(g) for g in gs); eps = (h - len(gs) + 1) / (n - len(gs))
        return {"test": f"Kruskal-Wallis: index ~ {col}", "statistic": h, "p_value": p, "effect_size": eps, "effect_name": "epsilon-squared"}
    rows += [kw("Ownership Type"), kw("CMS Region")]
    for col in ["Ownership Type", "CMS Region"]:
        ct = pd.crosstab(d[col], d["performance_category"]); chi, p, dof, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1)))
        rows.append({"test": f"Chi-square: {col} x category", "statistic": chi, "p_value": p, "effect_size": v, "effect_name": "Cramer's V"})
    feat = fe.engineer_features(m).loc[d.index]
    for c in ["facility_age_years", "n_irf_in_state", "log_eligible_stays", "log_mspb_episodes", "log_function_cases"]:
        ok = feat[c].notna(); r, p = stats.spearmanr(feat.loc[ok, c], d.loc[ok, "quality_index"])
        rows.append({"test": f"Spearman: index ~ {c}", "statistic": r, "p_value": p, "effect_size": r, "effect_name": "rho"})
    for c in ["name_has_rehab", "name_encompass_chain"]:
        a, b = d.loc[feat[c] == 1, "quality_index"], d.loc[feat[c] == 0, "quality_index"]
        u, p = stats.mannwhitneyu(a, b); rows.append({"test": f"Mann-Whitney: index by {c} (n1={len(a)}, n0={len(b)})", "statistic": u, "p_value": p, "effect_size": 2 * u / (len(a) * len(b)) - 1, "effect_name": "rank-biserial"})
    res = pd.DataFrame(rows)
    # pairwise ownership (Bonferroni), groups with n>=20
    pw = []; gs = [g for g in d["Ownership Type"].unique() if (d["Ownership Type"] == g).sum() >= 20]
    from itertools import combinations
    pairs = list(combinations(gs, 2))
    for a, b in pairs:
        x, y = d.loc[d["Ownership Type"] == a, "quality_index"], d.loc[d["Ownership Type"] == b, "quality_index"]
        u, p = stats.mannwhitneyu(x, y); pw.append({"pair": f"{a} vs {b}", "median_a": x.median(), "median_b": y.median(), "p_raw": p, "p_bonferroni": min(1, p * len(pairs))})
    pw = pd.DataFrame(pw); res.to_csv(TAB / "diagnostic_tests.csv", index=False); pw.to_csv(TAB / "ownership_pairwise.csv", index=False)
    return res, pw


# ------------------------------------------------------------------ 5. models
def _feature_sets(m, df):
    ok = df["quality_index"].notna(); X = fe.engineer_features(m)[ok]
    XB = X.copy()
    for c in HELD_OUT_MEASURES: XB[c] = m.loc[ok, c]
    y = df.loc[ok, "performance_category"].map({c: i for i, c in enumerate(CLASS_ORDER)}).astype(int)
    return X, XB, y


def run_experiment(X, y, tag, save=True):
    Xtr, Xte, ytr, yte = mt.split(X, y)
    fitted, cv_tbl = mt.tune_models(Xtr, ytr)
    hyb, vw = mt.build_hybrids(fitted, cv_tbl, Xtr, ytr)
    allm = {**fitted, **hyb}; probas = {k: v.predict_proba(Xte) for k, v in allm.items()}
    rows = []
    for k, pr in probas.items():
        r = ev.evaluate(yte, pr); lo, hi = ev.bootstrap_ci(yte, pr)
        r["F1 (macro) 95% CI"] = f"[{lo:.3f}, {hi:.3f}]"
        r["CV macro-F1 (train)"] = cv_tbl.set_index("model")["cv_macro_f1"].get(k, np.nan) if k in cv_tbl["model"].values else np.nan
        rows.append({"Model": k, **r})
    res = pd.DataFrame(rows)
    for k in ["Hybrid: Soft Voting", "Hybrid: Weighted Voting", "Hybrid: Stacking"]:
        res.loc[res.Model == k, "CV macro-F1 (train)"] = mt.cv_macro_f1(allm[k], Xtr, ytr)
    res.to_csv(TAB / f"model_comparison_{tag}.csv", index=False); cv_tbl.to_csv(TAB / f"tuning_{tag}.csv", index=False)
    if save:
        for k, v in allm.items(): joblib.dump(v, MOD / f"{tag}_{k.replace(' ', '_').replace(':', '')}.joblib")
    return dict(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, models=allm, probas=probas, res=res, cv=cv_tbl, vw=vw)


def stage_models(m, df):
    X, XB, y = _feature_sets(m, df)
    dist = y.map(dict(enumerate(CLASS_ORDER))).value_counts().reindex(CLASS_ORDER)
    A = run_experiment(X, y, "A_characteristics")
    B = run_experiment(XB, y, "B_plus_heldout_quality", save=False)
    # leakage demonstration (NOT a result): classifying from the very measures that define the index
    P = fe.direction_adjusted_percentiles(m[list(INDEX_MEASURES)])[df["quality_index"].notna()]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    leak = cross_val_score(__import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([("i", __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer()), ("rf", RandomForestClassifier(200, random_state=SEED))]), P, y, cv=mt.CV, scoring="accuracy").mean()
    ev.plot_confusion({k: v for k, v in A["probas"].items()}, A["yte"], FIG / "fig11_confusion_matrices.png")
    ev.plot_roc_pr(A["probas"], A["yte"], FIG / "fig12_roc_curves.png", FIG / "fig13_pr_curves.png")
    ev.plot_comparison(A["res"], FIG / "fig14_model_comparison.png")
    ev.plot_calibration(A["probas"], A["yte"], FIG / "fig15_calibration_high.png")
    ev.plot_comparison(B["res"], FIG / "fig16_model_comparison_setB.png")
    # RF feature importance
    rf = A["models"]["Random Forest"]; names = ex.feature_names(rf)
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5)); imp.plot.barh(ax=ax, color="#4c72b0"); ax.set(xlabel="Impurity-based importance", title="Random Forest feature importance (set A)"); _save(fig, "fig17_rf_importance.png")
    return A, B, dist, leak


# ------------------------------------------------------------------ 6. SHAP
def stage_shap(A, df):
    import shap
    xgb = A["models"]["XGBoost"]; Xte = A["Xte"]; yte = A["yte"].values
    sv, Xt, names = ex.shap_values(xgb, Xte)
    imp = ex.global_importance(sv, names); imp.to_csv(TAB / "shap_importance.csv")
    ex.plot_importance(imp, FIG / "fig18_shap_importance.png")
    ex.plot_beeswarm(sv, Xt, names, 2, FIG / "fig19_shap_beeswarm_high.png"); ex.plot_beeswarm(sv, Xt, names, 0, FIG / "fig20_shap_beeswarm_low.png")
    base = shap.TreeExplainer(xgb.named_steps["clf"]).expected_value
    pr = A["probas"]["XGBoost"]; local = []
    for k, c in enumerate(CLASS_ORDER):
        cand = np.where((pr.argmax(1) == k) & (yte == k))[0]
        i = cand[np.argmax(pr[cand, k])]; ccn = df.loc[Xte.index[i], CCN]; nm = df.loc[Xte.index[i], "Provider Name"]
        ex.plot_local(sv, Xt, names, i, k, base[k], FIG / f"fig21_shap_local_{c.lower()}.png", f"{c}-category facility (CCN {ccn}); predicted P={pr[i, k]:.2f}")
        contrib = pd.Series(sv[i, :, k], index=names).sort_values()
        local.append({"category": c, "CCN": ccn, "facility": nm, "pred_prob": float(pr[i, k]),
                      "top_positive": {n: round(float(v), 3) for n, v in contrib.tail(3)[::-1].items()},
                      "top_negative": {n: round(float(v), 3) for n, v in contrib.head(3).items()}})
    json.dump(local, open(TAB / "shap_local.json", "w"), indent=1)
    return imp, local
