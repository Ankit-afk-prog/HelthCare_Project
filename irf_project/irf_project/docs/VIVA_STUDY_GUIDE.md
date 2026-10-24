# Viva study guide (learn the ideas; answer in your own words)
1 Why this problem? Public IRF quality data lets us study how facility traits relate to reported quality.
2 Why these datasets? Approved CMS IRF files; General = facility traits, Provider = quality measures; same CCN.
3 Why two files? Traits (ownership, region) and quality measures are stored separately.
4 How merged? Provider is long (65 rows/facility): pivot to one row per CCN, then 1:1 left join; assert 1,222 rows.
5 Target? Tercile category (Low/Medium/High) of a Quality Performance Index - project-specific, not CMS ratings.
6 Why not use CMS labels? They exist only per measure, and not for overall performance.
7 Index steps? Direction-adjust (lower-is-better flipped) -> percentile -> weighted mean of available measures (>=8/11) -> terciles.
8 Weights? Equal as main; domain-balanced and PCA as sensitivity (Spearman 0.905 / 0.942 vs equal).
9 Missing values? Index averages available measures; <8 excluded (99); features: median imputation inside CV.
10 Categorical? One-hot (ownership, region). 11 Outliers? Reported by IQR, kept (genuine facility values).
12 Class imbalance? Terciles balanced, so no SMOTE; SMOTE would be train-only.
13 Why Logistic Regression? Simple, interpretable baseline. 14 Random Forest? Handles non-linearity/mixed features, robust.
15 XGBoost? Boosted trees, strong on tabular data. 16 Decision tree? Interpretable baseline.
17 Why ensemble? Averages different models' errors; tested, not assumed better.
18 What is SHAP? Splits a prediction into per-feature contributions (Shapley values). 19 Why XAI? Transparency; SHAP explains the model, not causes.
20 Data leakage? Information from the target leaking into features; inflates results. Index measures are excluded from features (demo: using them gives 0.79 CV accuracy).
21 Why dangerous? Results look great but fail on new data and mislead.
22 Metric? Macro-F1 (balanced classes, 3-class) plus ROC-AUC/PR-AUC; accuracy vs 0.333 chance.
23 Validation? Stratified 80/20 hold-out + 5-fold CV tuning, preprocessing inside pipeline.
24 Macro vs weighted? Macro averages classes equally. 25 Multiclass ROC-AUC? One-vs-rest, macro-averaged.
26 Results modest? Say so honestly: characteristics alone are weak; see fact sheet; do not over-claim.
27 Did hybrid win? Look at the fact sheet per experiment and compare with CIs.
28 Novelty? Index + sensitivity, leakage-safe two designs, hybrid vs baselines, SHAP.
29 Clinical decision-making? No: facility-level quality analytics only. 30 Limitations/future? Different reporting periods, relative terciles, no case-mix, associations not causal; add Conditions file, multi-year data.
