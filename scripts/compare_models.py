"""
Comparaison Régression Logistique vs XGBoost.

Objectif : vérifier si un modèle non-linéaire capture davantage de
signal, ou si la relation linéaire (logistique) est suffisante.
Sortie : courbe ROC comparative + importance variables XGBoost.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from src import config as C
from src.modeling import build_design_matrix
from src.viz_utils import _save

# XGBoost peut être absent — gérer gracieusement
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    print("[warn] xgboost non installé — `pip install xgboost`")
    XGB_AVAILABLE = False


def main():
    df = pd.read_csv(C.DATA_MODEL)
    X, y = build_design_matrix(df, C.MODEL_FEATURES,
                               target="bat_activee_bin")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y,
        random_state=C.RANDOM_STATE)

    # === Logistique
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=C.RANDOM_STATE)
    lr.fit(X_tr_s, y_tr)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_te, lr_prob)
    auc_lr = auc(fpr_lr, tpr_lr)

    results = {"logistic": {"auc_test": float(auc_lr)}}

    # === XGBoost
    if XGB_AVAILABLE:
        # Class imbalance handling
        pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=C.RANDOM_STATE,
        )
        xgb.fit(X_tr, y_tr)
        xgb_prob = xgb.predict_proba(X_te)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(y_te, xgb_prob)
        auc_xgb = auc(fpr_xgb, tpr_xgb)
        cv_xgb = cross_val_score(xgb, X_tr, y_tr, cv=5,
                                 scoring="roc_auc")
        print(f"XGBoost AUC test = {auc_xgb:.3f}")
        print(f"XGBoost AUC CV   = {cv_xgb.mean():.3f} "
              f"± {cv_xgb.std():.3f}")
        results["xgboost"] = {
            "auc_test": float(auc_xgb),
            "auc_cv_mean": float(cv_xgb.mean()),
            "auc_cv_std": float(cv_xgb.std()),
        }
        # Comparaison ROC
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr_lr, tpr_lr, lw=2.5, color="#457B9D",
                label=f"Logistique (AUC={auc_lr:.3f})")
        ax.plot(fpr_xgb, tpr_xgb, lw=2.5, color="#E63946",
                label=f"XGBoost (AUC={auc_xgb:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("1 - Spécificité")
        ax.set_ylabel("Sensibilité")
        ax.set_title("ROC — Logistique vs XGBoost")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")
        fig.tight_layout()
        _save(fig, "09_roc_comparison")

        # Feature importance
        imp = pd.Series(xgb.feature_importances_,
                        index=X.columns).sort_values()
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(imp))))
        ax.barh(imp.index, imp.values, color="#2A9D8F",
                edgecolor="black", linewidth=0.5)
        ax.set_title("Importance XGBoost (gain)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        _save(fig, "10_xgb_importance")

    out = C.TBL_DIR / "model_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] → {out}")


if __name__ == "__main__":
    main()
