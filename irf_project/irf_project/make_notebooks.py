"""Generates the 11 project notebooks (run once; then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path
SETUP = """import sys, json; sys.path.insert(0, '..')
import pandas as pd, numpy as np
from IPython.display import Image, display
from src import preprocessing as pp, feature_engineering as fe, pipeline as P
from src.config import *
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)
def show(name, w=800): display(Image(filename=str(FIG / name), width=w))"""
NB = {}
NB["01_data_loading"] = [("md", "# 01 Data loading\nLoad the two approved CMS IRF files (September 2026 refresh)."),
 ("code", "gen, prov = pp.load_raw()\nprint('General Information:', gen.shape); print('Provider Data      :', prov.shape)\ngen.head(3)"), ("code", "prov.head(5)")]
NB["02_data_understanding"] = [("md", "# 02 Data understanding\nStructure, missingness, duplicates, join key and cardinality."),
 ("code", "gen, prov = pp.load_raw()\nins = pd.concat([pp.inspect_table(gen,'General'), pp.inspect_table(prov,'Provider')]); ins"),
 ("code", "print('Duplicate rows  general/provider:', gen.duplicated().sum(), prov.duplicated().sum())\nprint('Unique CCN        general/provider:', gen[CCN].nunique(), prov[CCN].nunique())\nprint('Provider CCN all in General:', prov[CCN].isin(gen[CCN]).all(), '| General CCN all in Provider:', gen[CCN].isin(prov[CCN]).all())\nprint('Rows per CCN in Provider:', prov.groupby(CCN).size().unique())\nprint('Duplicate (CCN, Measure Code):', prov.duplicated([CCN,'Measure Code']).sum())\nprint('=> General:Provider is one-to-many (1 : 65).')"),
 ("code", "prov['Measure Code'].str.extract(r'^(I_\\d{3}_\\d{2})')[0].value_counts().sort_index()"),
 ("code", "prov.groupby('Measure Date Range').size().head(3); prov[['Start Date','End Date']].drop_duplicates()")]
NB["03_data_cleaning"] = [("md", "# 03 Data cleaning\nScore is text: numbers (some with thousands-commas), 'Not Available', or CMS comparison labels."),
 ("code", "gen, prov = pp.load_raw(); p = pp.clean_provider(prov)\nprint('Rows with non-numeric score:', p.score_num.isna().sum())\np.score_label.value_counts()"),
 ("code", "p.groupby('Footnote', dropna=False).size()  # footnote meanings: CMS dictionary Table 10"),
 ("code", "wide = pp.pivot_provider(p); pp.invalid_value_report(wide)"),
 ("md", "Outliers are **reported, not removed** - extreme rates/ratios can be genuine facility observations."),
 ("code", "m = pd.read_csv(PROC/'irf_facilities_merged.csv'); pp.outlier_report(m, list(INDEX_MEASURES))")]
NB["04_dataset_integration"] = [("md", "# 04 Dataset integration\nPivot the long Provider table to one row per CCN, then 1:1 left-join onto General Information (validated)."),
 ("code", "gen, prov = pp.load_raw(); g = pp.clean_general(gen); p = pp.clean_provider(prov); wide = pp.pivot_provider(p)\nmerged, info = pp.merge_facilities(g, wide); info['provider_long_rows'] = len(prov)\ninfo"),
 ("code", "print('Naive long merge would give', len(g.merge(prov[[CCN,'Measure Code']], on=CCN)), 'rows (65 per facility) - avoided by pivoting.')\nprint('Merged facility table:', merged.shape, '| unique CCN =', merged[CCN].nunique())"),
 ("code", "merged[['Provider Name','State','Ownership Type']+list(INDEX_MEASURES)[:4]].head()")]
NB["05_eda"] = [("md", "# 05 Exploratory data analysis\nFigures are produced by `P.stage_eda`; each cell notes what the figure shows."),
 ("code", SETUP.split('\n')[0] and "m = pd.read_csv(PROC/'irf_facilities_merged.csv', dtype={CCN:str}, parse_dates=['Certification Date'])\ndf = pd.read_csv(PROC/'facility_quality_index.csv', dtype={CCN:str})\nPm = fe.direction_adjusted_percentiles(m[list(INDEX_MEASURES)])\ndesc, own, reg, miss = P.stage_eda(m, df, Pm)\ndesc"),
 ("code", "show('fig01_missingness.png', 650); miss.round(1)"), ("code", "show('fig02_category_distribution.png', 450); show('fig06_index_distribution.png', 550)"),
 ("code", "show('fig03_geography.png'); own"), ("code", "show('fig04_ownership.png', 450); reg"), ("code", "show('fig05_measure_distributions.png')"),
 ("code", "show('fig07_correlation_heatmap.png', 750)"), ("code", "show('fig08_boxplots_ownership_region.png'); show('fig09_index_vs_volume.png', 550); show('fig10_measures_by_category.png')"),
 ("md", "Interpretations for the report are written from the *printed numbers* (see `outputs/tables`) - none are hard-coded.")]
NB["06_feature_engineering"] = [("md", "# 06 Feature engineering\nOnly facility-characteristic and volume features; **no index measure** is used as a feature (leakage control)."),
 ("code", "m = pd.read_csv(PROC/'irf_facilities_merged.csv', dtype={CCN:str}, parse_dates=['Certification Date'])\nX = fe.engineer_features(m); fe.FEATURE_DICTIONARY"),
 ("code", "X.describe(include='all').T"), ("code", "X.isna().mean().round(3)")]
NB["07_performance_index"] = [("md", "# 07 Quality Performance Index\ndirection adjustment -> percentile rank -> weighted mean over available measures (>= 8 of 11) -> tercile categories."),
 ("code", "m = pd.read_csv(PROC/'irf_facilities_merged.csv', dtype={CCN:str}, parse_dates=['Certification Date'])\ndf, Pm, sens, wtab, contrib, val, info = P.stage_index(m); info"),
 ("code", "wtab"), ("code", "contrib  # how strongly each measure's percentile tracks the final index"), ("md", "### Sensitivity to the weighting scheme"),
 ("code", "sens"), ("md", "### Descriptive comparison with CMS's own per-measure label (discharge to community). Partly circular (that measure is inside the index) -> descriptive only."),
 ("code", "val")]
NB["08_baseline_models"] = [("md", "# 08 Baseline models\nLogistic Regression, Decision Tree, Random Forest, XGBoost, plus a majority-class dummy. 80/20 stratified split; 5-fold CV grid search on the 80% (macro-F1). Set `RETRAIN=True` to re-run (~8 min)."),
 ("code", "RETRAIN = False\nif RETRAIN:\n    m = pd.read_csv(PROC/'irf_facilities_merged.csv', dtype={CCN:str}, parse_dates=['Certification Date'])\n    df = pd.read_csv(PROC/'facility_quality_index.csv', dtype={CCN:str}); df['performance_category']=pd.Categorical(df['performance_category'], CLASS_ORDER)\n    X, XB, y = P._feature_sets(m, df); A = P.run_experiment(X, y, 'A_characteristics')\npd.read_csv(TAB/'tuning_A_characteristics.csv')"),
 ("code", "t = pd.read_csv(TAB/'model_comparison_A_characteristics.csv'); t[t.Model.isin(['Majority-class baseline','Logistic Regression','Decision Tree','Random Forest','XGBoost'])].round(3)")]
NB["09_hybrid_model"] = [("md", "# 09 Proposed hybrid models\nSoft voting, CV-weighted soft voting, stacking - all built from the tuned LR + RF + XGBoost. The hybrid is *tested*, not assumed better."),
 ("code", "import json; print(json.load(open(OUT/'results.json'))['voting_weights'])\nt = pd.read_csv(TAB/'model_comparison_A_characteristics.csv'); t[t.Model.str.startswith('Hybrid') | t.Model.isin(['Logistic Regression','Random Forest','XGBoost'])].round(3)"),
 ("md", "### Set B (design 1 + held-out quality measures) and Design 2 (outcome index; care-process features)"),
 ("code", "pd.read_csv(TAB/'model_comparison_B_plus_heldout_quality.csv').round(3)"), ("code", "pd.read_csv(TAB/'model_comparison_C_outcome_index_process_features.csv').round(3)")]
NB["10_evaluation"] = [("md", "# 10 Evaluation\nAccuracy, macro precision/recall/F1, specificity, ROC-AUC (one-vs-rest macro), PR-AUC (macro), Brier, calibration, bootstrap CIs."),
 ("code", "pd.read_csv(TAB/'model_comparison_A_characteristics.csv').round(3).T"), ("code", "show('fig14_model_comparison.png', 700); show('fig11_confusion_matrices.png')"),
 ("code", "show('fig12_roc_curves.png', 500); show('fig13_pr_curves.png', 500); show('fig15_calibration_high.png', 450)"),
 ("md", "### Diagnostic statistics (associations, not causation)"), ("code", "pd.read_csv(TAB/'diagnostic_tests.csv').round(4)"), ("code", "pd.read_csv(TAB/'ownership_pairwise.csv').round(4)"),
 ("md", "### Leakage check"), ("code", "print('CV accuracy when the index measures themselves are used as features (leakage demo, NOT a result):', json.load(open(OUT/'results.json'))['leakage_demo_cv_accuracy_index_measures_as_features'])")]
NB["11_shap_explainability"] = [("md", "# 11 SHAP explainability\nTreeSHAP on XGBoost. SHAP explains the *model*, not the real world: no causal or clinical claims."),
 ("code", "pd.read_csv(TAB/'shap_importance.csv', index_col=0).round(4).head(15)"), ("code", "show('fig18_shap_importance.png', 600); show('fig19_shap_beeswarm_high.png', 650); show('fig20_shap_beeswarm_low.png', 650)"),
 ("code", "show('fig17_rf_importance.png', 550)"), ("md", "### Local explanations"),
 ("code", "local = json.load(open(TAB/'shap_local.json')); pd.DataFrame(local)"), ("code", "for c in ['low','medium','high']: show(f'fig21_shap_local_{c}.png', 650)"),
 ("md", "### Design 2"), ("code", "show('fig23_shap_importance_design2.png', 600); pd.read_csv(TAB/'shap_importance_design2.csv', index_col=0).round(4).head(12)")]
for name, cells in NB.items():
    nb = nbf.v4.new_notebook(); nb.cells = [nbf.v4.new_code_cell(SETUP)] + [nbf.v4.new_markdown_cell(c) if k == "md" else nbf.v4.new_code_cell(c) for k, c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, Path("notebooks") / f"{name}.ipynb")
print("notebooks written:", len(NB))
