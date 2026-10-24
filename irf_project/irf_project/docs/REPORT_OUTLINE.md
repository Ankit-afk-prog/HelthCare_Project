# Report outline (faculty's 19 sections) - what each section must contain
Write every section yourself. Facts/numbers: `RESULTS_FACT_SHEET.md`. Figures: `outputs/figures`.
1 Title page | 2 Abstract (problem, data, method, key results, one-line limitation) | 3 Introduction (why IRF quality matters; CMS QRP)
4 Problem statement (question, target = project-specific category, why analytics) | 5 Objectives (use the 7 in README)
6 Literature review (use YOUR uploaded review as base; verified sources so far: CMS IRF Data Dictionary v7.0; Edelstein et al. IRF ownership & QRP measures, PubMed 37907161; Ottenbacher et al. 30-day readmission after IRF, PMC4085109; Healthcare 2021, 9(4):486 hospital star-rating prediction; ML: Breiman 2001, Chen & Guestrin 2016, Lundberg & Lee 2017, Wolpert 1992, Dietterich 2000) - re-check every citation/DOI yourself
7 Dataset description (files, rows, columns, join, measures table, reporting periods, footnotes)
8 Preprocessing (long->wide pivot, text scores, commas, missing handling, outliers kept, leakage control, split)
9 Proposed methodology (index construction + weights + sensitivity; two designs; models; CV; hybrid)
10 Novelty (list exactly what is new/improved)
11 Implementation (tools, versions, repo structure)
12 Experimental results (tables for set A, B, Design 2)   13 Performance evaluation (metrics, CIs, confusion matrices, ROC/PR, calibration)
14 Visualisation and healthcare insights (each figure: what it shows + observation)
15 Comparison with baseline (majority class, individual models vs hybrid - report honestly)
16 Discussion (why weak/moderate signal; ownership/region patterns; SHAP; no causal claims)
17 Conclusion   18 Future scope   19 References
