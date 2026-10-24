"""Multiclass metrics, bootstrap CIs and publication-style plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from .config import CLASS_ORDER, SEED

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
K = len(CLASS_ORDER)


def specificity_macro(y, p):
    cm = confusion_matrix(y, p, labels=range(K)); out = []
    for i in range(K):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        out.append(tn / (tn + fp))
    return float(np.mean(out))


def evaluate(y, proba):
    """y: ints 0..K-1; proba: (n,K). Macro = unweighted mean over classes (all classes equally important)."""
    y = np.asarray(y); pred = proba.argmax(1); Y = label_binarize(y, classes=range(K))
    cm = confusion_matrix(y, pred, labels=range(K))
    return {
        "Accuracy": accuracy_score(y, pred), "Balanced Acc.": balanced_accuracy_score(y, pred),
        "Precision (macro)": precision_score(y, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y, pred, average="macro", zero_division=0),
        "Specificity (macro)": specificity_macro(y, pred),
        "F1 (macro)": f1_score(y, pred, average="macro"), "F1 (weighted)": f1_score(y, pred, average="weighted"),
        "ROC-AUC (OvR macro)": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "PR-AUC (macro)": float(np.mean([average_precision_score(Y[:, i], proba[:, i]) for i in range(K)])),
        "Brier (multiclass)": float(np.mean(((proba - Y) ** 2).sum(1))),
        "Low<->High errors (%)": 100 * (cm[0, 2] + cm[2, 0]) / len(y),   # ordinal 'two-step' errors
    }


def bootstrap_ci(y, proba, metric="F1 (macro)", n=1000):
    rng = np.random.default_rng(SEED); y = np.asarray(y); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < K: continue
        vals.append(evaluate(y[i], proba[i])[metric])
    return tuple(np.percentile(vals, [2.5, 97.5]))


def plot_confusion(models_proba, y, path):
    n = len(models_proba); cols = 3; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows)); axs = np.atleast_1d(axs).ravel()
    for ax, (nm, pr) in zip(axs, models_proba.items()):
        cm = confusion_matrix(y, pr.argmax(1), labels=range(K))
        ax.imshow(cm, cmap="Blues")
        for i in range(K):
            for j in range(K):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(K), CLASS_ORDER); ax.set_yticks(range(K), CLASS_ORDER)
        ax.set_xlabel("Predicted category"); ax.set_ylabel("True category"); ax.set_title(nm, fontsize=10)
    for ax in axs[n:]: ax.axis("off")
    fig.suptitle("Confusion matrices on the hold-out test set", y=1.0); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_roc_pr(models_proba, y, p_roc, p_pr):
    Y = label_binarize(y, classes=range(K)); grid = np.linspace(0, 1, 200)
    f1, a1 = plt.subplots(figsize=(6.2, 5)); f2, a2 = plt.subplots(figsize=(6.2, 5))
    for nm, pr in models_proba.items():
        if nm.startswith("Majority"): continue
        tprs = [np.interp(grid, *roc_curve(Y[:, i], pr[:, i])[:2]) for i in range(K)]
        auc = roc_auc_score(y, pr, multi_class="ovr", average="macro")
        a1.plot(grid, np.mean(tprs, 0), label=f"{nm} (AUC={auc:.3f})")
        prs = [np.interp(grid, precision_recall_curve(Y[:, i], pr[:, i])[1][::-1], precision_recall_curve(Y[:, i], pr[:, i])[0][::-1]) for i in range(K)]
        ap = np.mean([average_precision_score(Y[:, i], pr[:, i]) for i in range(K)])
        a2.plot(grid, np.mean(prs, 0), label=f"{nm} (AP={ap:.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=.8, label="Chance")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="Macro-averaged ROC curves (one-vs-rest), test set"); a1.legend(fontsize=7)
    a2.axhline(1 / K, color="k", ls="--", lw=.8, label="Chance (class prevalence)")
    a2.set(xlabel="Recall", ylabel="Precision", title="Macro-averaged precision-recall curves, test set"); a2.legend(fontsize=7)
    f1.tight_layout(); f1.savefig(p_roc); f2.tight_layout(); f2.savefig(p_pr); plt.close("all")


def plot_comparison(tbl, path, metrics=("Accuracy", "F1 (macro)", "ROC-AUC (OvR macro)", "PR-AUC (macro)")):
    t = tbl.set_index("Model")[list(metrics)]
    ax = t.plot.barh(figsize=(8.5, 5.5), width=.8); ax.invert_yaxis()
    ax.set(xlabel="Score (test set)", ylabel="", title="Model comparison on the hold-out test set"); ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); plt.tight_layout(); plt.savefig(path); plt.close()


def plot_calibration(models_proba, y, path, cls=2):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nm, pr in models_proba.items():
        if nm.startswith("Majority"): continue
        fr, mp = calibration_curve((np.asarray(y) == cls).astype(int), pr[:, cls], n_bins=6, strategy="quantile")
        ax.plot(mp, fr, marker="o", ms=3, label=nm)
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction",
        title=f"Calibration curve - class '{CLASS_ORDER[cls]}'"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
