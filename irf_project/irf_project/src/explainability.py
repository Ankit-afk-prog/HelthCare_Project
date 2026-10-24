"""SHAP explanations for the tree-based XGBoost model (model interpretation, not clinical inference)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from .config import CLASS_ORDER


def feature_names(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    return [n.split("__", 1)[1] for n in names]


def shap_values(pipe, X):
    """Returns (values[n, features, classes], transformed matrix, names)."""
    Xt = pipe.named_steps["prep"].transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
    return sv, Xt, feature_names(pipe)


def global_importance(sv, names):
    imp = pd.DataFrame(np.abs(sv).mean(0), index=names, columns=CLASS_ORDER)
    imp["Overall mean |SHAP|"] = imp.mean(1)
    return imp.sort_values("Overall mean |SHAP|", ascending=False)


def plot_importance(imp, path, top=12):
    t = imp.head(top)[CLASS_ORDER].iloc[::-1]
    ax = t.plot.barh(stacked=True, figsize=(7.5, 5), color=["#d95f02", "#7570b3", "#1b9e77"])
    ax.set(xlabel="Mean |SHAP value| (stacked by class)", title="Global feature importance (SHAP, XGBoost)"); plt.tight_layout(); plt.savefig(path); plt.close()


def plot_beeswarm(sv, Xt, names, cls, path):
    plt.figure(); shap.summary_plot(sv[:, :, cls], Xt, feature_names=names, show=False, max_display=12)
    plt.title(f"SHAP summary - class '{CLASS_ORDER[cls]}'"); plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close("all")


def plot_local(sv, Xt, names, i, cls, base, path, title):
    e = shap.Explanation(sv[i, :, cls], base_values=base, data=Xt[i], feature_names=names)
    plt.figure(); shap.plots.waterfall(e, max_display=8, show=False); plt.title(title, fontsize=9)
    plt.savefig(path, bbox_inches="tight"); plt.close("all")
