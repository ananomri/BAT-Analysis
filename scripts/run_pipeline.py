"""
Pipeline complet — point d'entrée du projet.

Usage:
    python -m scripts.run_pipeline

Étapes:
    1. Nettoyage des données
    2. Analyse descriptive (EDA)
    3. Analyses bivariées
    4. Régression logistique multivariée + sélection backward
    5. Évaluation prédictive (ROC, AUC, CV, confusion)
    6. Génération des figures et tables
"""
import json
import sys
from pathlib import Path

# Garder src/ importable depuis n'importe où
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config as C
from src.preprocessing import run_pipeline as clean_data
from src.stats_utils import (compare_continuous, compare_categorical,
                             odds_ratio_2x2)
from src.modeling import (build_design_matrix, compute_vif,
                          fit_logistic, or_table, backward_selection,
                          evaluate_model)
from src.viz_utils import (plot_target_balance, plot_numeric_distributions,
                           plot_categorical_associations,
                           plot_correlation_matrix, plot_forest_or,
                           plot_roc, plot_confusion_matrix,
                           plot_feature_importance)


def save_table(df: pd.DataFrame, name: str):
    C.TBL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TBL_DIR / f"{name}.csv", index=False)
    print(f"  → table sauvée : {name}.csv")


def main():
    # =========================================================
    # 1. CLEANING
    # =========================================================
    df = clean_data()

    # =========================================================
    # 2. EDA — VISUALISATIONS
    # =========================================================
    print("\n" + "=" * 60)
    print("ANALYSE EXPLORATOIRE")
    print("=" * 60)
    plot_target_balance(df)
    plot_numeric_distributions(df, ["age", "imc", "glycemie",
                                    "temp_ext", "heure_injection_h"])
    plot_categorical_associations(df, ["sexe", "diabete", "chimio_recente",
                                       "classe_temp", "saison",
                                       "cancer_grp", "etat_thyroidien"])
    plot_correlation_matrix(df, ["age", "poids_kg", "taille_m", "imc",
                                 "glycemie", "temp_ext",
                                 "heure_injection_h"])

    # =========================================================
    # 3. BIVARIÉ — VARIABLES CONTINUES
    # =========================================================
    print("\n" + "=" * 60)
    print("BIVARIÉ — VARIABLES CONTINUES (cas vs témoins)")
    print("=" * 60)
    cont_results = [compare_continuous(df, v)
                    for v in ["age", "poids_kg", "taille_m", "imc",
                              "glycemie", "temp_ext",
                              "heure_injection_h"]]
    cont_df = pd.DataFrame(cont_results)
    print(cont_df.to_string(index=False))
    save_table(cont_df, "bivariate_continuous")

    # =========================================================
    # 4. BIVARIÉ — VARIABLES CATÉGORIELLES
    # =========================================================
    print("\n" + "=" * 60)
    print("BIVARIÉ — VARIABLES CATÉGORIELLES")
    print("=" * 60)
    cat_results = [compare_categorical(df, v)
                   for v in ["sexe", "diabete", "chimio_recente",
                             "classe_temp", "saison", "cancer_grp",
                             "etat_thyroidien", "insuffisance_renale",
                             "imc_class", "age_class"]]
    cat_df = pd.DataFrame(cat_results)
    print(cat_df.to_string(index=False))
    save_table(cat_df, "bivariate_categorical")

    # =========================================================
    # 5. ODDS RATIO BIVARIÉS (variables binaires)
    # =========================================================
    print("\n" + "=" * 60)
    print("ODDS RATIO BIVARIÉS (variables 2×2)")
    print("=" * 60)
    or_results = [odds_ratio_2x2(df, v)
                  for v in ["sexe_f", "diabete_bin", "chimio_recente_bin",
                            "insuffisance_renale_bin"]]
    or_bivar_df = pd.DataFrame(or_results)
    print(or_bivar_df.to_string(index=False))
    save_table(or_bivar_df, "odds_ratios_bivariate")

    # =========================================================
    # 6. RÉGRESSION LOGISTIQUE MULTIVARIÉE
    # =========================================================
    print("\n" + "=" * 60)
    print("RÉGRESSION LOGISTIQUE MULTIVARIÉE")
    print("=" * 60)
    X, y = build_design_matrix(df, C.MODEL_FEATURES,
                               target="bat_activee_bin")
    print(f"Design matrix : {X.shape}")

    # VIF
    vif = compute_vif(X)
    print("\n--- VIF (détection colinéarité) ---")
    print(vif.to_string(index=False))
    save_table(vif, "vif")

    # Modèle complet
    full_model = fit_logistic(X, y)
    or_full = or_table(full_model)
    print("\n--- OR Modèle complet ---")
    print(or_full.to_string(index=False))
    save_table(or_full, "or_full_model")

    # Sélection backward
    final_model, kept_features = backward_selection(X, y, threshold=0.05)
    if final_model is not None:
        or_final = or_table(final_model)
        print("\n--- OR Modèle final (backward p<0.05) ---")
        print(or_final.to_string(index=False))
        save_table(or_final, "or_final_model")
        # Forest plot
        if len(or_final) > 0:
            plot_forest_or(or_final.copy(),
                           title="Forest plot — modèle final (multivarié)",
                           name="05_forest_final")

    # Forest plot complet
    or_full_plot = or_full[or_full["IC95_high"] < 50].copy()
    if len(or_full_plot) > 0:
        plot_forest_or(or_full_plot,
                       title="Forest plot — modèle complet",
                       name="05_forest_full")

    # =========================================================
    # 7. ÉVALUATION PRÉDICTIVE
    # =========================================================
    print("\n" + "=" * 60)
    print("ÉVALUATION PRÉDICTIVE (sklearn, CV-5)")
    print("=" * 60)
    eval_res = evaluate_model(X, y)
    print(f"AUC test     : {eval_res['auc']:.3f}")
    print(f"AUC CV (5)   : {eval_res['cv_auc_mean']:.3f} "
          f"± {eval_res['cv_auc_std']:.3f}")
    print(f"\nMatrice de confusion :\n{eval_res['confusion']}")
    print(f"\nMacro F1 : {eval_res['report']['macro avg']['f1-score']:.3f}")

    plot_roc(eval_res["fpr"], eval_res["tpr"], eval_res["auc"])
    plot_confusion_matrix(eval_res["confusion"])
    plot_feature_importance(eval_res["coefficients"],
                            title="Coefficients standardisés — Log. Reg.")

    # Sauvegarde JSON métriques
    metrics_path = C.TBL_DIR / "model_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "auc_test": float(eval_res["auc"]),
            "auc_cv_mean": float(eval_res["cv_auc_mean"]),
            "auc_cv_std": float(eval_res["cv_auc_std"]),
            "kept_features_backward": kept_features,
            "n_train": int(len(y) * (1 - C.TEST_SIZE)),
            "n_test": int(len(y) * C.TEST_SIZE),
            "report": eval_res["report"],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[metrics] → {metrics_path}")

    # Sauvegarde dataset modélisé
    df_model = df.copy()
    df_model.to_csv(C.DATA_MODEL, index=False, encoding="utf-8")
    print(f"[save model dataset] → {C.DATA_MODEL}")

    print("\n" + "=" * 60)
    print("PIPELINE TERMINÉ ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
