"""Model construction, hyper-parameter tuning and the proposed hybrid (ensemble) models."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from .config import SEED, CLASS_ORDER

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def split(X, y, test_size=0.2):
    """Stratified hold-out test set (20%). The remaining 80% is used for 5-fold CV tuning/validation."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def make_preprocessor(X):
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("category", "str", "string")]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat)])


def model_specs():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
                                {"clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                          {"clf__max_depth": [3, 4, 5, 6], "clf__min_samples_leaf": [5, 10, 20]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
                          {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [3, 5, 10], "clf__max_features": ["sqrt", 0.5]}),
        "XGBoost": (XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", subsample=0.8,
                                  colsample_bytree=0.8, random_state=SEED, n_jobs=1, verbosity=0),
                    {"clf__n_estimators": [150, 300], "clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.1]}),
    }


def tune_models(Xtr, ytr, scoring="f1_macro"):
    """Grid-search each model inside a Pipeline(preprocess -> classifier) so imputation/scaling
    are re-fitted inside every CV fold (no leakage)."""
    fitted, rows = {}, []
    for name, (est, grid) in model_specs().items():
        pipe = Pipeline([("prep", make_preprocessor(Xtr)), ("clf", est)])
        gs = GridSearchCV(pipe, grid, cv=CV, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        fitted[name] = gs.best_estimator_
        rows.append({"model": name, "cv_macro_f1": gs.best_score_,
                     "cv_std": gs.cv_results_["std_test_score"][gs.best_index_],
                     "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}})
    dummy = Pipeline([("prep", make_preprocessor(Xtr)), ("clf", DummyClassifier(strategy="prior"))]).fit(Xtr, ytr)
    fitted["Majority-class baseline"] = dummy
    return fitted, pd.DataFrame(rows)


def build_hybrids(fitted, cv_table, Xtr, ytr):
    """Proposed hybrid models built from the TUNED base models (LR, RF, XGB).
    1. Soft voting (equal weights)   2. Weighted soft voting (weights = CV macro-F1)   3. Stacking (LR meta-learner)."""
    base = ["Logistic Regression", "Random Forest", "XGBoost"]
    est = [(n, clone(fitted[n])) for n in base]
    w = cv_table.set_index("model").loc[base, "cv_macro_f1"].values
    hyb = {
        "Hybrid: Soft Voting": VotingClassifier(est, voting="soft"),
        "Hybrid: Weighted Voting": VotingClassifier(est, voting="soft", weights=list(w / w.sum())),
        "Hybrid: Stacking": StackingClassifier(est, final_estimator=LogisticRegression(max_iter=2000), cv=5,
                                               stack_method="predict_proba", n_jobs=1),
    }
    for m in hyb.values():
        m.fit(Xtr, ytr)
    return hyb, dict(zip(base, w / w.sum()))


def cv_macro_f1(model, Xtr, ytr):
    """5-fold out-of-fold macro-F1 on the training portion (for fair comparison incl. hybrids)."""
    pred = cross_val_predict(clone(model), Xtr, ytr, cv=CV, n_jobs=1)
    return f1_score(ytr, pred, average="macro")
