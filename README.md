# 🧬 BAT Activation Analysis — Étude cas-témoin sur la graisse brune en TEP/TDM ¹⁸FDG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Statut](https://img.shields.io/badge/statut-actif-success)]()
[![Vercel](https://img.shields.io/badge/deploy-Vercel-black)](https://bat-analysis-pfe-rouav2.vercel.app/)

> **Étude observationnelle rétrospective cas-témoin** réalisée par **Anan Omri** pour aider son amie radiologue (**Roua Zarouk**) dans le cadre de son Projet de Fin d'Études (PFE).
> les facteurs cliniques, biologiques et environnementaux associés à
> l'**activation de la graisse brune (BAT)** lors des examens de TEP/TDM
> au ¹⁸F-FDG, afin de **réduire les faux positifs diagnostiques** liés
> aux hyperfixations physiologiques.


## 🔗 Live Demo
Accédez au dashboard en ligne : [**https://bat-analysis-pfe-rouav2.vercel.app/**](https://bat-analysis-pfe-rouav2.vercel.app/)

---

## 📋 Sommaire
- [Contexte clinique](#-contexte-clinique)
- [Données](#-données)
- [Architecture du projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Méthodologie statistique](#-méthodologie-statistique)
- [Résultats principaux](#-résultats-principaux)
- [Visualisations](#-visualisations)
- [Limites & perspectives](#-limites--perspectives)
- [Citation](#-citation)

---

## 🩺 Contexte clinique

La **graisse brune (Brown Adipose Tissue, BAT)** est un tissu adipeux
thermogène hautement métabolique. En conditions de **stress froid**, elle
peut fixer le **¹⁸F-FDG** de manière physiologique, créant des foyers
hypermétaboliques sur l'imagerie TEP/TDM qui **miment des lésions tumorales**.

**Impact clinique** :
- Faux positifs diagnostiques en oncologie
- Risque de surdiagnostic / surtraitement
- Reprises d'examens, surcoût

**Objectif de l'étude** : identifier les facteurs **indépendants**
associés à l'activation BAT pour :
1. Mieux interpréter les examens
2. Adapter les conditions d'examen (réchauffement préalable)
3. Construire un score prédictif de risque

---

## 📊 Données

| Champ                    | Description                            |
| ------------------------ | -------------------------------------- |
| **Type d'étude**         | Cas-témoin observationnelle, rétrospective |
| **Effectif total**       | 333 patients TEP/TDM ¹⁸F-FDG          |
| **Cas (BAT activée)**    | 103 (30.9 %)                          |
| **Témoins**              | 230 (69.1 %)                          |
| **Variables**            | 23 → 17 après anonymisation           |
| **Période**              | Examens TEP-TDM consécutifs           |

### Variables analysées
- **Démographiques** : âge, sexe
- **Anthropométriques** : poids, taille, IMC
- **Biologiques** : glycémie, état thyroïdien, insuffisance rénale
- **Cliniques** : diabète, chimiothérapie récente, type de cancer, indication
- **Environnementales** : température extérieure, classe de température, saison
- **Procédurales** : heure de l'injection, temps d'acquisition

> 🔒 **Anonymisation** : les colonnes `ID`, `Nom`, `Prénom`, `Date`,
> `Numéro de téléphone` sont supprimées avant toute analyse.

---

## 🗂 Architecture du projet

```
bat-analysis/
├── data/
│   ├── raw/                      # CSV brut (gitignored)
│   └── processed/                # Données nettoyées
├── src/
│   ├── config.py                 # Constantes, paths, paramètres
│   ├── preprocessing.py          # Pipeline nettoyage + anonymisation
│   ├── stats_utils.py            # Tests bivariés intelligents
│   ├── modeling.py               # Régression logistique + VIF + sélection
│   └── viz_utils.py              # Charte graphique + plots
├── scripts/
│   ├── run_pipeline.py           # Pipeline complet end-to-end
│   └── compare_models.py         # Logistique vs XGBoost
├── reports/
│   ├── figures/                  # Plots PNG haute résolution
│   ├── tables/                   # Résultats statistiques (CSV/JSON)
│   └── final_report.md           # Rapport scientifique complet
├── notebooks/                    # Exploration interactive (optionnel)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le repo
git clone https://github.com/ananomri/BAT-Analysis.git
cd bat-analysis

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer le CSV brut dans data/raw/
cp /chemin/vers/votre_export.csv data/raw/bat_dataset_raw.csv
```

---

## 🚀 Utilisation

### Pipeline complet (recommandé)
```bash
python -m scripts.run_pipeline
```
Génère :
- Dataset nettoyé : `data/processed/bat_dataset_clean.csv`
- 9 figures dans `reports/figures/`
- Tables statistiques dans `reports/tables/`
- Métriques JSON du modèle

### Comparaison ML
```bash
python -m scripts.compare_models
```

### Pipeline modulaire (Python)
```python
from src.preprocessing import run_pipeline
from src.stats_utils import compare_continuous, odds_ratio_2x2
from src.modeling import fit_logistic, or_table

df = run_pipeline()
print(compare_continuous(df, "age"))
```

---

## 🔬 Méthodologie statistique

### 1. Nettoyage & préparation
- Détection des **outliers physiologiques** (ex. IMC>60 → recalcul depuis P/T²)
- Correction des typos (`dysthyroidie` / `dysthroidie` → `Dysthyroïdie`)
- Imputation **médiane** pour les rares NA numériques
- Regroupement des cancers en **8 classes** (puis 6 pour la modélisation)
- Encodage binaire 0/1 pour Oui/Non
- Conversion `Heure injection` en heure décimale

### 2. Analyse bivariée (sélection automatique des tests)
| Type de variable      | Test 1er choix         | Bascule auto si...               |
| --------------------- | ---------------------- | -------------------------------- |
| Continue ~ normale    | t-test Student/Welch   | Shapiro p<0.05 → Mann-Whitney    |
| Catégorielle (2×k)    | Chi² Pearson           | Effectifs attendus <5 → Fisher exact / G-test |

### 3. Régression logistique multivariée
- **Variables candidates** : retenues si p < 0.20 en bivariée (Hosmer-Lemeshow)
- **Détection colinéarité** : VIF > 5 = suspect, > 10 = exclusion
- **Sélection backward** : élimination itérative p > 0.05
- **Robustesse numérique** : cascade Newton → BFGS → L2 régularisé
  pour gérer la séparation parfaite des effectifs rares
- **Résultats** : OR + IC 95% + p-values

### 4. Évaluation prédictive
- **Train/Test split stratifié** 75/25
- **Validation croisée 5-fold** sur ROC-AUC
- **Métriques** : AUC, sensibilité, spécificité, F1 macro
- **Class weight balanced** (déséquilibre 30/70)
- **Comparaison** Logistique vs XGBoost (vérifier si non-linéarité utile)

---

## 📈 Résultats principaux

### Modèle final (régression logistique multivariée, sélection backward)

| Variable               | OR    | IC 95 %        | p-value | Signif |
| ---------------------- | ----- | -------------- | ------- | ------ |
| **Âge (par année)**    | 0.955 | [0.940 ; 0.969]| < 0.001 | ***    |
| **Glycémie**           | 1.721 | [1.370 ; 2.161]| < 0.001 | ***    |
| **Cancer du sein**     | 2.494 | [1.125 ; 5.529]| 0.024   | *      |

### Performance prédictive
- **AUC test** : **0.768**
- **AUC CV-5** : **0.701 ± 0.073**
- **Sensibilité** : 0.81 | **Spécificité** : 0.55

### Comparaison modèles
| Modèle              | AUC test | AUC CV-5     | Interprétabilité |
| ------------------- | -------- | ------------ | ---------------- |
| Régression Log.     | **0.768**| 0.701±0.073  | ⭐⭐⭐⭐⭐         |
| XGBoost             | 0.756    | 0.713±0.049  | ⭐⭐              |

> Conclusion : la **relation est essentiellement linéaire**. Le modèle
> logistique offre la meilleure performance ET reste cliniquement
> interprétable (OR directs). XGBoost n'apporte pas de gain ici.

---

## 🎨 Visualisations

Toutes les figures sont en haute résolution (200 dpi) dans `reports/figures/` :

| Fichier                          | Description                                  |
| -------------------------------- | -------------------------------------------- |
| `01_target_balance.png`          | Répartition cas/témoins                      |
| `02_numeric_distributions.png`   | KDE + boxplots stratifiés par variable       |
| `03_categorical_associations.png`| % activation BAT par modalité catégorielle   |
| `04_correlation_matrix.png`      | Heatmap Spearman                             |
| `05_forest_full.png`             | Forest plot — modèle complet                 |
| `05_forest_final.png`            | Forest plot — modèle final (backward)        |
| `06_roc_curve.png`               | Courbe ROC + AUC                             |
| `07_confusion_matrix.png`        | Matrice de confusion + Sens/Spec             |
| `08_feature_importance.png`      | Coefficients standardisés                    |
| `09_roc_comparison.png`          | Logistique vs XGBoost                        |
| `10_xgb_importance.png`          | Importance XGBoost (gain)                    |

---

## 🩺 Interprétation clinique

### Facteurs **indépendants** d'activation BAT
1. **Âge jeune** : OR=0.955/an → un patient de 30 ans a ~3× plus de risque
   qu'un patient de 60 ans, indépendamment des autres facteurs. Cohérent
   avec la régression involutive de la BAT après l'enfance/adolescence.

2. **Hyperglycémie** : OR=1.72 par mmol/L. **Paradoxe résolu** : en
   bivarié le diabète apparaît protecteur (OR=0.27) car les diabétiques
   sont plus âgés. Après ajustement sur l'âge, **c'est l'hyperglycémie
   aiguë qui devient un facteur de risque** — possiblement reflet d'un
   stress métabolique aigu (jeûne incomplet, stress préanalytique).

3. **Cancer du sein** : OR=2.49. Possiblement lié à la prédominance
   féminine et à la jeunesse relative de cette cohorte ; à confirmer
   sur cohorte indépendante.

### Recommandations pratiques
| Patient à risque ↑                | Mesure préventive                     |
| --------------------------------- | ------------------------------------- |
| Femme jeune (<40 ans)             | Pièce de repos préchauffée (24-26°C)  |
| Examen en saison froide           | Couverture chauffante 30 min pré-inj  |
| Glycémie élevée au prélèvement    | Vérifier jeûne strict ≥ 6 h           |
| Cancer du sein                    | Surveillance accrue sus-claviculaire  |

---

## ⚠️ Limites & perspectives

### Limites
- **Effectif modeste** (n=333) → IC larges sur sous-groupes
- **Étude monocentrique** → généralisabilité à valider
- **Variables manquantes** : pas de SUVmax, pas de T° corporelle réelle,
  pas de cortisolémie, pas d'IMC viscéral (TDM)
- **Pas de validation externe** sur cohorte indépendante
- **Sous-groupes rares** (Été n=8, dysthyroïdies n=5) — puissance limitée

### Perspectives
- [ ] Validation externe sur une seconde cohorte
- [ ] Ajout de la **SUVmax BAT** (variable continue → modèle linéaire)
- [ ] Calcul d'un **score de risque clinique** (nomogramme)
- [ ] Étude prospective avec randomisation du **réchauffement préalable**
- [ ] Modèle deep learning sur les **images PET-CT brutes**
- [ ] Déploiement d'une **API Flask/FastAPI** pour score temps réel

---

## 🛠 Stack technique

| Couche               | Outils                                          |
| -------------------- | ----------------------------------------------- |
| Data wrangling       | `pandas`, `numpy`                              |
| Statistiques         | `scipy.stats`, `statsmodels`                   |
| ML                   | `scikit-learn`, `xgboost`                      |
| Visualisation        | `matplotlib`, `seaborn`                        |
| Reproductibilité     | `requirements.txt`, seed=42                    |

---

## 📚 Citation

Si vous utilisez ce code dans une publication :
```bibtex
@misc{bat_activation_2026,
  title  = {BAT Activation Analysis: a case-control study of brown adipose tissue activation on 18F-FDG PET/CT},
  author = {Anan Omri},
  year   = {2026},
  url    = {https://github.com/ananomri/BAT-Analysis.git}
}
```

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE).

> **Note éthique** : les données patient ont été **anonymisées** avant
> tout traitement statistique conformément aux exigences RGPD et au cadre
> éthique de la recherche biomédicale (loi Jardé / déclaration d'Helsinki).
