"""
Configuration centrale du projet BAT Activation Analysis.
Tous les paths, constantes et paramètres statistiques sont définis ici.
"""
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "bat_dataset_latest.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "bat_dataset_clean.csv"
DATA_MODEL = ROOT / "data" / "processed" / "bat_dataset_model.csv"

FIG_DIR = ROOT / "reports" / "figures"
TBL_DIR = ROOT / "reports" / "tables"
REPORT_MD = ROOT / "reports" / "final_report.md"

# ============================================================
# ENCODING
# ============================================================
RAW_ENCODING = "utf-8-sig"   # CSV exporté avec BOM (UTF-8)

# ============================================================
# COLONNES À SUPPRIMER (anonymisation RGPD/éthique médicale)
# ============================================================
PII_COLUMNS = ["ID", "Nom", "Prénom", "Numéro de téléphone", "Date", "Prnom", "Numro de tlphone"]

# ============================================================
# RENOMMAGE STANDARDISÉ (codes propres pour Python)
# ============================================================
RENAME_MAP = {
    "Âge": "age",
    "ge": "age",
    "Sexe": "sexe",
    "Poids (kg)": "poids_kg",
    "Taille (m)": "taille_m",
    "IMC": "imc",
    "Glycémie": "glycemie",
    "Glycmie": "glycemie",
    "Diabète": "diabete",
    "Diabte": "diabete",
    "Chimio récente": "chimio_recente",
    "Chimio rcente": "chimio_recente",
    "BAT activée": "bat_activee",
    "BAT active": "bat_activee",
    "T(°C)": "temp_ext",
    "T(C)": "temp_ext",
    "classe température": "classe_temp",
    "classe temprature": "classe_temp",
    "Saison": "saison",
    "Cancer": "cancer",
    "Indication": "indication",
    "état throidien": "etat_thyroidien",
    "tat throidien": "etat_thyroidien",
    "etat throidien": "etat_thyroidien",
    "Insuffisance rénale": "insuffisance_renale",
    "Insuffisance rnale": "insuffisance_renale",
    "Heure de l’injection": "heure_injection",
    "Heure de linjection": "heure_injection",
    "Heure de l'injection": "heure_injection",
    "Temps d’acquisition": "temps_acquisition",
    "Temps dacquisition": "temps_acquisition",
    "Temps d'acquisition": "temps_acquisition",
    "Antécédant d'activation du BAT": "antecedent_bat",
}

# ============================================================
# VARIABLES PAR TYPE
# ============================================================
TARGET = "bat_activee"

NUMERIC_VARS = ["age", "poids_kg", "taille_m", "imc", "glycemie", "temp_ext",
                "heure_injection_h"]

CATEGORICAL_VARS = ["sexe", "diabete", "chimio_recente", "classe_temp",
                    "saison", "cancer_grp", "etat_thyroidien",
                    "insuffisance_renale"]

# Variables candidates pour la régression logistique multivariée.
# Choix raisonné :
#  - On garde `classe_temp` (3 niveaux) plutôt que `saison` (4 + Été n=8)
#    OU temp_ext brute — éviter la triple-redondance.
#  - `imc` plutôt que poids+taille (dérivée, plus interprétable).
#  - `cancer_grp_reduced` avec catégories ayant ≥ 15 observations
#    pour stabilité numérique de la régression.
MODEL_FEATURES = ["age", "glycemie", "cancer_hodgkin", "antecedent_bat"]

# ============================================================
# SEUILS & PARAMÈTRES STATISTIQUES
# ============================================================
ALPHA = 0.05
RANDOM_STATE = 42

# Seuil mini-effectif par cellule pour autoriser le Chi²
# (sinon → Fisher exact, ou regroupement)
CHI2_MIN_EXPECTED = 5

# Seuil de p-value pour passage en multivariée
P_THRESHOLD_BIVARIATE = 0.20  # convention médicale (Hosmer-Lemeshow)

# Seuil VIF pour suspicion de colinéarité
VIF_THRESHOLD = 5.0

# Train/test split
TEST_SIZE = 0.25

# Regroupement cancers (réduit la sparsité)
CANCER_GROUPING = {
    "Lymphome": "Hématologique",
    "Lymphome Hodgkin": "Hématologique",
    "Cancer du sein": "Sein",
    "Cancer du poumon": "Thoracique",
    "Cancer ORL": "ORL/Cavum",
    "UCNT cavum": "ORL/Cavum",
    "Cancer colorectal": "Digestif",
    "Cancer estomac": "Digestif",
    "Cancer pancréas": "Digestif",
    "Cancer oesophage": "Digestif",
    "Cancer duodénum": "Digestif",
    "Cancer ovaire": "Gynécologique",
    "Cancer du col": "Gynécologique",
    "Cancer endomètre": "Gynécologique",
    "Cancer utérus": "Gynécologique",
    "Cancer testicules": "Génito-urinaire",
    "Cancer testicule": "Génito-urinaire",
    "Cancer rein": "Génito-urinaire",
    "Mélanome": "Cutané/Sarcome",
    "Sarcomes": "Cutané/Sarcome",
    "Cancer thyroïde": "Endocrinien",
    "Neuroblastome": "Autre",
    "Syndrome néphrotique": "Autre",
    "Autre": "Autre",
}
