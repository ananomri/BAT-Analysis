"""
Pipeline de nettoyage des données BAT.

Responsabilités:
 - Anonymisation (suppression PII)
 - Standardisation des noms de colonnes
 - Correction des typos (dysthyroidie, ...)
 - Détection et correction des outliers physiologiques
 - Encodage binaire des Oui/Non
 - Conversion temporelle (heure d'injection → numérique)
 - Regroupement des cancers en classes
"""
import pandas as pd
import numpy as np
from . import config as C


def load_raw() -> pd.DataFrame:
    """Charge le CSV brut avec l'encodage approprié."""
    df = pd.read_csv(C.DATA_RAW, encoding=C.RAW_ENCODING)
    # Nettoyer les caractères apostrophe Windows-1252 (\x92) dans les noms
    df.columns = [c.replace("\x92", "'") for c in df.columns]
    return df


def anonymize(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes identifiantes (RGPD / éthique médicale)."""
    cols_to_drop = [c for c in C.PII_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def standardize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes en snake_case sans accents."""
    # Gérer les noms 'Heure de l'injection' / 'Temps d'acquisition' qui
    # ont l'apostrophe propre (après nettoyage de \x92)
    extra = {
        "Heure de l'injection": "heure_injection",
        "Temps d'acquisition": "temps_acquisition",
    }
    full_map = {**C.RENAME_MAP, **extra}
    return df.rename(columns=full_map)


def fix_thyroid_typos(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne `dysthyroidie` et `dysthroidie` → `Dysthyroïdie`."""
    df = df.copy()
    df["etat_thyroidien"] = (df["etat_thyroidien"]
                             .replace({"dysthyroidie": "Dysthyroïdie",
                                       "dysthroidie": "Dysthyroïdie"}))
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes Oui/Non en 0/1 (suffixées _bin)."""
    df = df.copy()
    binary_cols = ["diabete", "chimio_recente", "bat_activee",
                   "insuffisance_renale"]
    for c in binary_cols:
        df[f"{c}_bin"] = df[c].map({"Oui": 1, "Non": 0})
    # Sexe → 1 si Femme, 0 si Homme
    df["sexe_f"] = df["sexe"].map({"F": 1, "H": 0})
    return df


def parse_injection_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme 'HH:MM:SS' en heure décimale (ex: 09:30 → 9.5).
    Hypothèse clinique : un effet circadien possible (BAT plus active
    le matin froid). On garde aussi l'heure brute pour réf.
    """
    df = df.copy()
    def to_hours(s):
        if pd.isna(s):
            return np.nan
        try:
            h, m, *_ = str(s).split(":")
            return int(h) + int(m) / 60
        except (ValueError, AttributeError):
            return np.nan
    df["heure_injection_h"] = df["heure_injection"].apply(to_hours)
    return df


def fix_imc_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcule l'IMC si valeur physiologiquement impossible (>60 ou <10).
    L'IMC est dérivé : poids / taille² — donc on peut le recomputer.
    """
    df = df.copy()
    mask_aberrant = (df["imc"] > 60) | (df["imc"] < 10)
    n_fix = mask_aberrant.sum()
    if n_fix > 0:
        print(f"[clean] {n_fix} valeur(s) IMC aberrante(s) → recalcul "
              "depuis poids/taille")
        df.loc[mask_aberrant, "imc"] = (
            df.loc[mask_aberrant, "poids_kg"]
            / (df.loc[mask_aberrant, "taille_m"] ** 2)
        )
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputation médiane pour les rares NA numériques."""
    df = df.copy()
    for col in ["poids_kg", "imc", "glycemie", "heure_injection_h"]:
        if col in df.columns:
            n_na = df[col].isna().sum()
            if n_na > 0:
                med = df[col].median()
                df[col] = df[col].fillna(med)
                print(f"[impute] {col}: {n_na} NA → médiane {med:.2f}")
    return df


def group_cancers(df: pd.DataFrame) -> pd.DataFrame:
    """Réduit la sparsité en regroupant les cancers en classes."""
    df = df.copy()
    df["cancer_grp"] = df["cancer"].map(C.CANCER_GROUPING).fillna("Autre")
    # Pour la modélisation : regroupement plus agressif (≥15 obs/cat)
    small_groups = ["Génito-urinaire", "Endocrinien"]
    df["cancer_grp_reduced"] = df["cancer_grp"].replace(
        {g: "Autre" for g in small_groups})
    return df


def add_imc_class(df: pd.DataFrame) -> pd.DataFrame:
    """Classes OMS d'IMC — utile pour stratification clinique."""
    df = df.copy()
    bins = [0, 18.5, 25, 30, 100]
    labels = ["Maigreur", "Normal", "Surpoids", "Obésité"]
    df["imc_class"] = pd.cut(df["imc"], bins=bins, labels=labels,
                             include_lowest=True)
    return df


def add_age_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tranches d'âge cliniques."""
    df = df.copy()
    bins = [0, 18, 40, 60, 120]
    labels = ["<18", "18-40", "40-60", ">60"]
    df["age_class"] = pd.cut(df["age"], bins=bins, labels=labels,
                             include_lowest=True)
    return df


def run_pipeline() -> pd.DataFrame:
    """Pipeline complet de nettoyage. Retourne le dataframe propre."""
    print("=" * 60)
    print("PIPELINE DE NETTOYAGE BAT")
    print("=" * 60)
    df = load_raw()
    print(f"[load] {len(df)} patients × {df.shape[1]} colonnes")
    df = anonymize(df)
    print(f"[anon] PII supprimées → {df.shape[1]} colonnes")
    df = standardize_names(df)
    df = fix_thyroid_typos(df)
    df = parse_injection_time(df)
    df = fix_imc_outliers(df)
    df = impute_missing(df)
    df = group_cancers(df)
    df = add_imc_class(df)
    df = add_age_class(df)
    df = encode_binary(df)
    # Sauvegarde
    C.DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.DATA_CLEAN, index=False, encoding="utf-8")
    print(f"[save] → {C.DATA_CLEAN}")
    print(f"[done] Dataset final: {df.shape}")
    return df


if __name__ == "__main__":
    run_pipeline()
