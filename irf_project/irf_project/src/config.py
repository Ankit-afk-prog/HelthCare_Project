"""Central configuration: paths, seed, and the verified quality-measure catalogue.

Measure meanings/directions come from the official CMS IRF QRP Data Dictionary
(v7.0, September 2026), Tables 8-9 and Appendix A.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG, TAB, MOD = OUT / "figures", OUT / "tables", OUT / "models"
for _p in (PROC, FIG, TAB, MOD):
    _p.mkdir(parents=True, exist_ok=True)

GENERAL_FILE = RAW / "irf_general_information.csv"
PROVIDER_FILE = RAW / "irf_provider_data.csv"
CCN = "CMS Certification Number (CCN)"
SEED = 42
REFERENCE_YEAR = 2026            # data release: September 2026
MIN_MEASURES = 8                 # facility needs >= 8 of the 11 index measures

# code -> (readable name, better direction, domain)
INDEX_MEASURES = {
    "I_015_01_SIR":         ("C. difficile infection SIR",                       "lower",  "Safety"),
    "I_013_01_OBS_RATE":    ("Falls with major injury (%)",                      "lower",  "Safety"),
    "I_022_01_ADJ_RATE":    ("New/worsened pressure ulcers, risk-adj. (%)",      "lower",  "Safety"),
    "I_017_01_PPR_PD_RSRR": ("Preventable 30-day post-discharge readmission (RSRR)", "lower", "Readmission & community return"),
    "I_018_01_PPR_WI_RSRR": ("Preventable within-stay readmission (RSRR)",       "lower",  "Readmission & community return"),
    "I_019_02_DTC_RS_RATE": ("Discharge to community, risk-std. (%)",            "higher", "Readmission & community return"),
    "I_026_01_OBS_RATE":    ("Discharge function score met/exceeded (%)",       "higher", "Functional outcome"),
    "I_021_01_OBS_RATE":    ("Drug regimen review with follow-up (%)",          "higher", "Care processes"),
    "I_024_01_OBS_RATE":    ("Med list to next care setting (%)",               "higher", "Care processes"),
    "I_025_02_OBS_RATE":    ("Med list to patient/caregiver (%)",               "higher", "Care processes"),
    "I_020_01_MSPB_SCORE":  ("Medicare spending per beneficiary (ratio)",       "lower",  "Cost efficiency"),
}
SHORT = {k: v[0] for k, v in INDEX_MEASURES.items()}

# Quality measures deliberately kept OUT of the index (used only in the ablation feature set B)
HELD_OUT_MEASURES = ["I_016_01_OBS_RATE", "I_006_01_SIR"]
# Volume / size variables (denominators) -> facility-size features
VOLUME_CODES = ["I_017_01_PPR_PD_VOLUME", "I_020_01_MSPB_NUMB", "I_026_01_DENOMINATOR"]
# CMS per-measure comparative labels: used only to *validate* the index, never as features
COMP_CODES = ["I_017_01_PPR_PD_COMP_PERF", "I_018_01_PPR_WI_COMP_PERF", "I_019_02_DTC_COMP_PERF",
              "I_015_01_COMP_PERF", "I_006_01_COMP_PERF"]
CLASS_ORDER = ["Low", "Medium", "High"]
