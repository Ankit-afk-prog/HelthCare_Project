"""Streamlit dashboard.  Run from project root:  streamlit run dashboard/app.py
(Run `python run_pipeline.py` first so that data/processed and outputs/ exist.)"""
import json
from pathlib import Path
import pandas as pd, streamlit as st, plotly.express as px

ROOT = Path(__file__).resolve().parents[1]; PROC, OUT = ROOT / "data/processed", ROOT / "outputs"
FIG, TAB = OUT / "figures", OUT / "tables"
ORDER = ["Low", "Medium", "High"]; COL = {"Low": "#d95f02", "Medium": "#7570b3", "High": "#1b9e77"}
st.set_page_config(page_title="IRF Quality Analysis", layout="wide")

@st.cache_data
def load():
    d = pd.read_csv(PROC / "facility_quality_index.csv", dtype={"CMS Certification Number (CCN)": str})
    return d
df = load(); scored = df[df["quality_index"].notna()].copy()
page = st.sidebar.radio("Page", ["1 Overview", "2 Facility Quality Analysis", "3 Machine Learning", "4 Explainable AI", "5 Insights"])
st.sidebar.caption("Categories are project-specific analytical performance categories, NOT official CMS ratings. "
                   "This is a healthcare-analytics project, not a clinical decision-support tool.")

if page.startswith("1"):
    st.title("Inpatient Rehabilitation Facility (IRF) Quality Analysis")
    c = st.columns(4); c[0].metric("Facilities (CMS file)", len(df)); c[1].metric("Facilities scored", len(scored))
    c[2].metric("Mean index (0-100)", f"{scored.quality_index.mean():.1f}"); c[3].metric("Excluded (<8 measures)", len(df) - len(scored))
    a, b = st.columns(2)
    cnt = scored.performance_category.value_counts().reindex(ORDER).reset_index(); cnt.columns = ["Category", "Facilities"]
    a.plotly_chart(px.bar(cnt, x="Category", y="Facilities", color="Category", color_discrete_map=COL, title="Performance-category distribution"), use_container_width=True)
    b.plotly_chart(px.histogram(scored, x="quality_index", nbins=30, title="Quality Performance Index distribution"), use_container_width=True)
    st.subheader("Index construction (measure weights & direction)"); st.dataframe(pd.read_csv(TAB / "index_weights.csv"), use_container_width=True)
    st.subheader("Descriptive statistics of the raw measures"); st.dataframe(pd.read_csv(TAB / "descriptive_measures.csv", index_col=0).round(2), use_container_width=True)

elif page.startswith("2"):
    st.title("Facility Quality Analysis")
    s = st.sidebar.multiselect("State", sorted(scored.State.unique())); o = st.sidebar.multiselect("Ownership", sorted(scored["Ownership Type"].unique()))
    r = st.sidebar.multiselect("CMS Region", sorted(scored["CMS Region"].unique())); k = st.sidebar.multiselect("Performance category", ORDER, default=ORDER)
    f = scored[scored.performance_category.isin(k)]
    if s: f = f[f.State.isin(s)]
    if o: f = f[f["Ownership Type"].isin(o)]
    if r: f = f[f["CMS Region"].isin(r)]
    st.write(f"**{len(f)}** facilities match the filters.")
    a, b = st.columns(2)
    a.plotly_chart(px.box(f, x="Ownership Type", y="quality_index", color="Ownership Type", title="Index by ownership"), use_container_width=True)
    b.plotly_chart(px.box(f, x=f["CMS Region"].astype(str), y="quality_index", title="Index by CMS Region", labels={"x": "CMS Region"}), use_container_width=True)
    g = f.groupby("State").agg(facilities=("quality_index", "size"), mean_index=("quality_index", "mean")).reset_index()
    st.plotly_chart(px.choropleth(g, locations="State", locationmode="USA-states", color="mean_index", scope="usa", hover_data=["facilities"], title="Mean index by state (states with few IRFs are noisy)"), use_container_width=True)
    st.dataframe(f.sort_values("quality_index", ascending=False)[["Provider Name", "State", "Ownership Type", "CMS Region", "quality_index", "performance_category", "n_measures_available"]].round(2), use_container_width=True)
    st.subheader("Association tests (not causal)"); st.dataframe(pd.read_csv(TAB / "diagnostic_tests.csv").round(4), use_container_width=True)

elif page.startswith("3"):
    st.title("Machine-learning classification")
    ex = {"Design 1 - set A (facility characteristics only)": "A_characteristics", "Design 1 - set B (+ held-out quality measures)": "B_plus_heldout_quality",
          "Design 2 (outcome index; process + characteristic features)": "C_outcome_index_process_features"}
    ch = st.selectbox("Experiment", list(ex)); f = TAB / f"model_comparison_{ex[ch]}.csv"
    if f.exists():
        t = pd.read_csv(f); st.dataframe(t.round(3), use_container_width=True)
        m = st.multiselect("Metrics", ["Accuracy", "F1 (macro)", "ROC-AUC (OvR macro)", "PR-AUC (macro)"], default=["F1 (macro)", "ROC-AUC (OvR macro)"])
        st.plotly_chart(px.bar(t.melt(id_vars="Model", value_vars=m), x="value", y="Model", color="variable", barmode="group", orientation="h"), use_container_width=True)
    if ex[ch] == "A_characteristics":
        st.image(str(FIG / "fig11_confusion_matrices.png")); a, b = st.columns(2); a.image(str(FIG / "fig12_roc_curves.png")); b.image(str(FIG / "fig13_pr_curves.png"))
        st.image(str(FIG / "fig15_calibration_high.png"), width=420)

elif page.startswith("4"):
    st.title("Explainable AI (SHAP) - model interpretation, not causation")
    imp = pd.read_csv(TAB / "shap_importance.csv", index_col=0)
    st.image(str(FIG / "fig18_shap_importance.png")); st.dataframe(imp.round(4).head(15), use_container_width=True)
    st.image(str(FIG / "fig19_shap_beeswarm_high.png"), width=620)
    local = json.load(open(TAB / "shap_local.json")); pick = st.selectbox("Representative facility", [f"{l['category']}: {l['facility']} ({l['CCN']})" for l in local])
    l = local[[f"{x['category']}: {x['facility']} ({x['CCN']})" for x in local].index(pick)]
    st.write(f"Predicted category **{l['category']}** (probability {l['pred_prob']:.2f}).")
    a, b = st.columns(2); a.write("Largest positive contributors"); a.json(l["top_positive"]); b.write("Largest negative contributors"); b.json(l["top_negative"])
    st.image(str(FIG / f"fig21_shap_local_{l['category'].lower()}.png"))

else:
    st.title("Insights (computed live from the data)")
    t = pd.read_csv(TAB / "diagnostic_tests.csv"); own = pd.read_csv(TAB / "ownership_summary.csv", index_col=0); reg = pd.read_csv(TAB / "region_summary.csv", index_col=0)
    st.markdown(f"- The dataset shows **{len(scored)}** scored facilities; median index differs by ownership: " + ", ".join(f"{i}: {v:.1f}" for i, v in own["median"].items()) + ".")
    st.markdown(f"- Highest / lowest median index by CMS Region: **Region {reg['median'].idxmax()}** ({reg['median'].max():.1f}) / **Region {reg['median'].idxmin()}** ({reg['median'].min():.1f}).")
    for _, r in t.iterrows(): st.markdown(f"- {r['test']}: statistic={r['statistic']:.3f}, p={r['p_value']:.4g}, {r['effect_name']}={r['effect_size']:.3f}")
    st.info("These are associations in cross-sectional public-reporting data, not causal effects. Differences may reflect case-mix, size, "
            "reporting periods or chance. Suggested use: prioritising areas for operational/quality review, never clinical decisions.")
    st.caption("See docs/ and the final report for the full discussion and limitations.")
