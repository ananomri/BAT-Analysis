"""
Modélisation : régression logistique multivariée + alternative XGBoost.

Objectif clinique : identifier les facteurs INDÉPENDANTS associés
à l'activation BAT, et fournir un modèle prédictif avec interprétation
en Odds Ratio + IC95%.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             classification_report)
from . import config as C


# ============================================================
# PRÉPARATION DESIGN MATRIX
# ============================================================
def build_design_matrix(df: pd.DataFrame, features: list,
                        target: str = "bat_activee_bin"):
    """
    Encode les variables catégorielles en one-hot (drop_first pour
    éviter la dummy variable trap).
    Retourne (X, y) prêts pour la modélisation.
    Si target n'est pas dans df, y est None.
    """
    X = df[features].copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True,
                       dtype=float)
    
    y = None
    if target in df.columns:
        y = df[target].astype(int)
    return X, y


# ============================================================
# VIF — DÉTECTION DE COLINÉARITÉ
# ============================================================
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    VIF > 5 = colinéarité suspectée.
    VIF > 10 = colinéarité forte → à exclure.
    """
    X_const = sm.add_constant(X)
    vifs = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        try:
            v = variance_inflation_factor(X_const.values, i)
        except Exception:
            v = np.nan
        vifs.append({"variable": col, "VIF": round(v, 2)})
    return pd.DataFrame(vifs).sort_values("VIF", ascending=False)


# ============================================================
# RÉGRESSION LOGISTIQUE MULTIVARIÉE (statsmodels)
# ============================================================
def fit_logistic(X: pd.DataFrame, y: pd.Series) -> sm.Logit:
    """
    Régression logistique avec statsmodels (pour OR + IC + p-values).
    Cascade de robustesse :
      1. Newton (par défaut, rapide)
      2. BFGS (plus stable si séparation quasi-parfaite)
      3. Régularisation L2 légère (dernier recours)
    """
    X_const = sm.add_constant(X)
    try:
        return sm.Logit(y, X_const).fit(disp=False, maxiter=200)
    except (np.linalg.LinAlgError, Exception):
        pass
    try:
        return sm.Logit(y, X_const).fit(disp=False, maxiter=500,
                                         method="bfgs")
    except (np.linalg.LinAlgError, Exception):
        pass
    # Dernier recours : régularisation L2 légère pour casser la séparation
    print("[warn] Régularisation L2 activée (séparation détectée)")
    return sm.Logit(y, X_const).fit_regularized(
        alpha=0.05, L1_wt=0, disp=False, maxiter=500)


def or_table(model: sm.Logit) -> pd.DataFrame:
    """Convertit les coefficients en OR + IC95% + p-values."""
    params = model.params
    conf = model.conf_int()
    conf.columns = ["IC95_low", "IC95_high"]
    out = pd.concat([params, conf], axis=1)
    out.columns = ["coef", "IC95_low", "IC95_high"]
    out["OR"] = np.exp(out["coef"])
    out["IC95_low"] = np.exp(out["IC95_low"])
    out["IC95_high"] = np.exp(out["IC95_high"])
    out["p_value"] = model.pvalues
    out = out.drop(columns=["coef"])
    out["signif"] = out["p_value"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01
        else "*" if p < 0.05 else "ns")
    out = out.reset_index().rename(columns={"index": "variable"})
    out = out[out["variable"] != "const"]
    return out[["variable", "OR", "IC95_low", "IC95_high",
                "p_value", "signif"]].round(4)


# ============================================================
# SÉLECTION DE MODÈLE (backward stepwise sur p-values)
# ============================================================
def backward_selection(X: pd.DataFrame, y: pd.Series,
                       threshold: float = 0.05) -> tuple:
    """Élimine itérativement les variables non-significatives."""
    features = list(X.columns)
    while True:
        if not features:
            break
        Xs = X[features]
        try:
            model = fit_logistic(Xs, y)
        except Exception:
            break
        pvals = model.pvalues.drop("const", errors="ignore")
        worst_p = pvals.max()
        if worst_p > threshold:
            worst = pvals.idxmax()
            features.remove(worst)
        else:
            break
    if features:
        final_model = fit_logistic(X[features], y)
        return final_model, features
    return None, []


# ============================================================
# ÉVALUATION PRÉDICTIVE (sklearn)
# ============================================================
def evaluate_model(X: pd.DataFrame, y: pd.Series):
    """Train/test split + métriques ROC/confusion."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE,
        stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=1000,
                             random_state=C.RANDOM_STATE,
                             class_weight="balanced")
    clf.fit(X_train_s, y_train)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    cv_auc = cross_val_score(clf, X_train_s, y_train, cv=5,
                             scoring="roc_auc")
    report = classification_report(y_test, y_pred, output_dict=True,
                                   zero_division=0)
    return {
        "model": clf,
        "scaler": scaler,
        "auc": auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "confusion": cm,
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std": cv_auc.std(),
        "report": report,
        "feature_names": list(X.columns),
        "coefficients": pd.Series(clf.coef_[0], index=X.columns),
    }
