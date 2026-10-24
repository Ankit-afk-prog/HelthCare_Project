# CA-III requirements checklist (from the faculty document) -> where it is covered

| # | Faculty requirement | Where in this project | Status |
|---|---|---|---|
| 1 | Use ONLY the dataset provided by Dr. Prerna on Moodle | CMS IRF General Information + Provider Data (Sept 2026) | **YOU must confirm these two files are the Moodle resource.** Earlier hospital-DB / country files were NOT used. |
| 1 | Mention dataset name, source, records, attributes, features | `docs/RESULTS_FACT_SHEET.md` (counts), `src/feature_engineering.FEATURE_DICTIONARY` | Numbers ready; write in your words |
| 2 | Clear problem statement, target, why analytics helps | Target = tercile category of the Quality Performance Index | Write yourself |
| 3 | Novel/improved approach (mandatory) | Hybrid ensemble (soft/weighted voting, stacking) + transparent performance index with weighting-sensitivity + SHAP + leakage-safe two-design comparison | Implemented; state novelty in your words; the hybrid's benefit must be reported as tested (see fact sheet) |
| 4 | Preprocessing explained and justified | notebooks 03, 04, 06; `src/preprocessing.py` | Done; justify in report |
| 4 | Class imbalance | Tercile classes are balanced (375/374/374) so no resampling needed | Done |
| 5 | Descriptive, diagnostic, predictive analytics | EDA (nb 05), tests (`diagnostic_tests.csv`), models (nb 08-09) | Done |
| 6 | Evaluation + baseline comparison | Accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, confusion matrices; baselines = majority class + individual models | Done |
| 6 | Regression / clustering metrics | Not applicable (classification) | N/A |
| 7 | Visualisations with insight | `outputs/figures` (24 figures), dashboard | Done; interpret each yourself |
| 8 | Tools and versions | `outputs/results.json` -> env, `requirements.txt` | Done |
| 9 | Report structure (19 sections) | Use the FACULTY structure (see REPORT_OUTLINE.md) | **You write it** |
| 10 | Original, plagiarism-free, **AI-free report** | - | **I did not write the report. Write it in your own words from the fact sheet.** |
| 11 | GitHub repo link, code, results | Whole `irf_project/` folder | **You must create the GitHub repo and add the link** |
| 12 | Presentation + live demo | Dashboard `streamlit run dashboard/app.py` | You prepare slides in your own words |
| 13 | Basic EDA not enough; real analysis | Models, hybrid, SHAP, tests | Done |
