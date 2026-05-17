"""
Utilitaires de visualisation — style cohérent pour publication.

Charte graphique :
 - Palette : froide (bleu) pour témoins, chaude (rouge) pour cas
 - Fond blanc, grille discrète, polices lisibles
 - Annotations de p-values et effectifs systématiques
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from . import config as C

# Charte graphique
PALETTE = {"Oui": "#E63946", "Non": "#457B9D"}   # cas / témoins
PALETTE_GROUP = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51",
                 "#8338EC", "#3A86FF", "#FB5607"]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def _save(fig, name: str):
    """Sauvegarde figure en PNG haute résolution."""
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_target_balance(df: pd.DataFrame):
    """Visualise la répartition cas/témoins."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[C.TARGET].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[k] for k in counts.index],
                  edgecolor="black", linewidth=0.8)
    total = counts.sum()
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"n = {n}\n({n/total*100:.1f} %)",
                ha="center", fontweight="bold")
    ax.set_title("Équilibre cas / témoins — Activation BAT")
    ax.set_ylabel("Effectif")
    ax.set_xlabel("BAT activée")
    ax.set_ylim(0, counts.max() * 1.2)
    return _save(fig, "01_target_balance")


def plot_numeric_distributions(df: pd.DataFrame, vars_: list):
    """Grille de KDE+boxplot par variable continue, stratifiée."""
    n = len(vars_)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, v in enumerate(vars_):
        # KDE
        for grp in ["Non", "Oui"]:
            sns.kdeplot(df.loc[df[C.TARGET] == grp, v],
                        ax=axes[i, 0], fill=True, alpha=0.4,
                        color=PALETTE[grp], label=f"BAT = {grp}")
        axes[i, 0].set_title(f"Distribution — {v}")
        axes[i, 0].legend()
        # Boxplot
        sns.boxplot(data=df, x=C.TARGET, y=v, ax=axes[i, 1],
                    palette=PALETTE, hue=C.TARGET, legend=False)
        sns.stripplot(data=df, x=C.TARGET, y=v, ax=axes[i, 1],
                      color="black", alpha=0.3, size=2)
        axes[i, 1].set_title(f"Comparaison — {v}")
    fig.suptitle("Variables continues vs activation BAT",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    return _save(fig, "02_numeric_distributions")


def plot_categorical_associations(df: pd.DataFrame, vars_: list):
    """Barres groupées avec pourcentages d'activation BAT par modalité."""
    n = len(vars_)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    for i, v in enumerate(vars_):
        ct = pd.crosstab(df[v], df[C.TARGET], normalize="index") * 100
        ct = ct.reindex(columns=["Non", "Oui"])
        ct.plot(kind="bar", stacked=True, ax=axes[i],
                color=[PALETTE["Non"], PALETTE["Oui"]],
                edgecolor="black", linewidth=0.6)
        axes[i].set_title(f"% activation BAT par {v}")
        axes[i].set_ylabel("Proportion (%)")
        axes[i].set_xlabel("")
        axes[i].legend(title="BAT", loc="upper right")
        axes[i].tick_params(axis="x", rotation=30)
        # Ajout effectifs sur barres
        counts = df[v].value_counts().reindex(ct.index)
        for j, n_mod in enumerate(counts.values):
            axes[i].text(j, 102, f"n={n_mod}", ha="center", fontsize=8)
    for k in range(len(vars_), len(axes)):
        axes[k].axis("off")
    fig.tight_layout()
    return _save(fig, "03_categorical_associations")


def plot_correlation_matrix(df: pd.DataFrame, cols: list):
    """Heatmap des corrélations Spearman (robuste aux outliers)."""
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, fmt=".2f",
                cbar_kws={"label": "ρ Spearman"},
                linewidths=0.5, ax=ax)
    ax.set_title("Corrélations Spearman (variables continues)")
    fig.tight_layout()
    return _save(fig, "04_correlation_matrix")


def plot_forest_or(or_df: pd.DataFrame, title: str = "Forest plot — OR",
                   name: str = "05_forest_plot"):
    """Forest plot des Odds Ratio avec IC95%."""
    or_df = or_df.sort_values("OR")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(or_df))))
    y = np.arange(len(or_df))
    ax.errorbar(or_df["OR"], y,
                xerr=[or_df["OR"] - or_df["IC95_low"],
                      or_df["IC95_high"] - or_df["OR"]],
                fmt="o", color="#1D3557", capsize=4, markersize=7,
                ecolor="#457B9D", elinewidth=1.5)
    ax.axvline(1, color="red", linestyle="--", alpha=0.6, label="OR=1")
    ax.set_yticks(y)
    ax.set_yticklabels(or_df["variable"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (échelle log)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, name)


def plot_roc(fpr, tpr, auc_val, name: str = "06_roc_curve"):
    """Courbe ROC publication-ready."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#E63946", lw=2.5,
            label=f"Modèle (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Hasard")
    ax.set_xlabel("Taux de faux positifs (1 - Spécificité)")
    ax.set_ylabel("Taux de vrais positifs (Sensibilité)")
    ax.set_title("Courbe ROC — Modèle de régression logistique")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    return _save(fig, name)


def plot_confusion_matrix(cm, name: str = "07_confusion_matrix"):
    """Matrice de confusion annotée (Sensibilité/Spécificité)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred. Non", "Pred. Oui"],
                yticklabels=["Réel Non", "Réel Oui"],
                cbar=False, linewidths=1, linecolor="white", ax=ax)
    ax.set_title(f"Matrice de confusion\n"
                 f"Sensibilité={sens:.2f} | Spécificité={spec:.2f}")
    fig.tight_layout()
    return _save(fig, name)


def plot_feature_importance(importances: pd.Series,
                            title="Importance des variables",
                            name="08_feature_importance"):
    """Barplot horizontal d'importance (coefs ou gain XGBoost)."""
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(importances))))
    colors = ["#E63946" if v > 0 else "#457B9D" for v in importances.values]
    ax.barh(importances.index, importances.values, color=colors,
            edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Coefficient / Importance")
    fig.tight_layout()
    return _save(fig, name)
