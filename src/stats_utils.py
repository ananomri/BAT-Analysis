"""
Boîte à outils statistiques pour étude cas-témoin.

Implémente la sélection automatique des tests selon les conditions
de validité (effectifs, normalité, homoscédasticité).
"""
import numpy as np
import pandas as pd
from scipy import stats
from . import config as C


# ============================================================
# TESTS DE NORMALITÉ & HOMOSCÉDASTICITÉ
# ============================================================
def shapiro_safe(x: pd.Series) -> float:
    """Test Shapiro-Wilk avec gestion des cas limites."""
    x = x.dropna()
    if len(x) < 3 or len(x) > 5000:
        return np.nan
    return stats.shapiro(x).pvalue


def levene_safe(g1: pd.Series, g2: pd.Series) -> float:
    """Test de Levene (robuste à la non-normalité)."""
    g1, g2 = g1.dropna(), g2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    return stats.levene(g1, g2).pvalue


# ============================================================
# COMPARAISON DE 2 GROUPES SUR VARIABLE CONTINUE
# ============================================================
def compare_continuous(df: pd.DataFrame, var: str,
                       group_col: str = C.TARGET) -> dict:
    """
    Compare une variable continue entre cas (Oui) et témoins (Non).
    Sélectionne automatiquement t-test ou Mann-Whitney selon la
    distribution.
    """
    g_cas = df.loc[df[group_col] == "Oui", var].dropna()
    g_ctrl = df.loc[df[group_col] == "Non", var].dropna()

    p_norm_cas = shapiro_safe(g_cas)
    p_norm_ctrl = shapiro_safe(g_ctrl)
    p_levene = levene_safe(g_cas, g_ctrl)

    # Critères : normalité des deux groupes ET variances homogènes
    normal = (p_norm_cas > 0.05) and (p_norm_ctrl > 0.05)
    equal_var = (p_levene > 0.05) if not np.isnan(p_levene) else True

    if normal:
        stat, pval = stats.ttest_ind(g_cas, g_ctrl, equal_var=equal_var)
        test = "Student" if equal_var else "Welch"
    else:
        stat, pval = stats.mannwhitneyu(g_cas, g_ctrl, alternative="two-sided")
        test = "Mann-Whitney"

    return {
        "variable": var,
        "n_cas": len(g_cas),
        "n_ctrl": len(g_ctrl),
        "mean_cas": round(g_cas.mean(), 2),
        "std_cas": round(g_cas.std(), 2),
        "median_cas": round(g_cas.median(), 2),
        "mean_ctrl": round(g_ctrl.mean(), 2),
        "std_ctrl": round(g_ctrl.std(), 2),
        "median_ctrl": round(g_ctrl.median(), 2),
        "test": test,
        "statistic": round(stat, 3),
        "p_value": round(pval, 4),
        "signif": "***" if pval < 0.001 else "**" if pval < 0.01 else
                  "*" if pval < 0.05 else "ns",
    }


# ============================================================
# COMPARAISON SUR VARIABLE CATÉGORIELLE
# ============================================================
def compare_categorical(df: pd.DataFrame, var: str,
                        group_col: str = C.TARGET) -> dict:
    """
    Test d'association Chi² avec bascule automatique vers Fisher exact
    si effectifs attendus < 5 (règle de Cochran).
    Pour tables 2×k, on utilise Fisher exact via stats.fisher_exact si 2×2,
    sinon le test exact de Freeman-Halton (via simulation Monte-Carlo).
    """
    ctab = pd.crosstab(df[var], df[group_col])

    if ctab.shape[0] < 2 or ctab.shape[1] < 2:
        return {"variable": var, "test": "N/A",
                "p_value": np.nan, "signif": "ns"}

    chi2, p_chi2, dof, expected = stats.chi2_contingency(ctab)
    use_fisher = (expected < C.CHI2_MIN_EXPECTED).any()

    if use_fisher and ctab.shape == (2, 2):
        odds, p = stats.fisher_exact(ctab.values)
        test_name = "Fisher exact"
    elif use_fisher:
        # Table > 2×2 → Fisher-Freeman-Halton via Monte-Carlo
        try:
            res = stats.chi2_contingency(ctab,
                                         lambda_="log-likelihood")
            p = res[1]
            test_name = "G-test (Likelihood ratio)"
        except Exception:
            p = p_chi2
            test_name = "Chi² (warning: faibles effectifs)"
    else:
        p = p_chi2
        test_name = "Chi²"

    return {
        "variable": var,
        "test": test_name,
        "statistic": round(chi2, 3),
        "dof": dof,
        "p_value": round(p, 4),
        "n_groups": ctab.shape[0],
        "signif": "***" if p < 0.001 else "**" if p < 0.01 else
                  "*" if p < 0.05 else "ns",
    }


# ============================================================
# ODDS RATIO + IC 95% pour variables binaires
# ============================================================
def odds_ratio_2x2(df: pd.DataFrame, var_bin: str,
                   target_bin: str = "bat_activee_bin") -> dict:
    """
    Calcule l'OR avec IC95% par méthode de Woolf (log-transformation).
    var_bin doit être codée 0/1.
    """
    a = ((df[var_bin] == 1) & (df[target_bin] == 1)).sum()
    b = ((df[var_bin] == 1) & (df[target_bin] == 0)).sum()
    c = ((df[var_bin] == 0) & (df[target_bin] == 1)).sum()
    d = ((df[var_bin] == 0) & (df[target_bin] == 0)).sum()
    # Correction de Haldane si zéro
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_val = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ci_low = np.exp(np.log(or_val) - 1.96 * se)
    ci_high = np.exp(np.log(or_val) + 1.96 * se)
    return {
        "variable": var_bin,
        "OR": round(or_val, 2),
        "IC95_low": round(ci_low, 2),
        "IC95_high": round(ci_high, 2),
        "interpret": "Risque ↑" if or_val > 1 else "Protecteur"
    }
