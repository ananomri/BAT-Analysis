"""
Analyse de causalité (Extension #3).
Utilise DoWhy pour estimer les effets de traitement (ATE) en contrôlant
les variables de confusion via un graphe acyclique dirigé (DAG).
"""
import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
from . import config as C


def prepare_causal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données pour DoWhy (besoin de types spécifiques)."""
    df_c = df.copy()
    # DoWhy préfère les booléens ou 0/1 pour le traitement/outcome
    df_c['bat_activee_bin'] = df_c['bat_activee_bin'].astype(bool)
    
    # Créer un traitement binaire pour la température (Froid vs non-froid)
    # Seuil arbitraire basé sur la distribution (ex: < 15°C)
    df_c['is_cold'] = df_c['temp_ext'] < 15
    
    # Créer un traitement binaire pour l'âge (Jeune vs vieux, seuil 40 ans)
    df_c['is_young'] = df_c['age'] < 40
    
    return df_c


def run_causal_inference(df: pd.DataFrame, 
                          treatment: str = 'is_young', 
                          outcome: str = 'bat_activee_bin'):
    """
    Exécute le pipeline DoWhy :
    1. Modéliser (DAG)
    2. Identifier (Identifier l'effet)
    3. Estimer (Estimation statistique)
    4. Réfuter (Vérifier la robustesse)
    """
    
    # Définition du DAG simplifié
    # On suppose que l'Age influence la Glycémie et l'IMC.
    # On suppose que la Saison/Température est exogène.
    causal_graph = """
    digraph {
        is_young -> bat_activee_bin;
        is_cold -> bat_activee_bin;
        age -> is_young;
        age -> glycemie;
        age -> imc;
        glycemie -> bat_activee_bin;
        imc -> bat_activee_bin;
        sexe_f -> bat_activee_bin;
        sexe_f -> age;
    }
    """
    
    # 1. Modéliser
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=causal_graph
    )
    
    # 2. Identifier
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    # 3. Estimer (via Propensity Score Matching ou Linear Regression)
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        target_units="ate"
    )
    
    print(f"*** Effet causal estimé ({treatment}) : {estimate.value:.4f}")
    
    # 4. Réfuter
    # Ajout d'une variable de confusion non observée (Placebo)
    refutation = model.refute_estimate(
        identified_estimand, 
        estimate,
        method_name="placebo_treatment_refuter"
    )
    
    return {
        "model": model,
        "estimand": identified_estimand,
        "estimate": estimate,
        "refutation": refutation
    }


def analyze_all_causal(df: pd.DataFrame):
    """Lance les analyses pour l'âge et le froid."""
    df_c = prepare_causal_data(df)
    
    results = {}
    
    print("\n--- Analyse : Effet de l'Age (<40 ans) ---")
    results['age'] = run_causal_inference(df_c, treatment='is_young')
    
    print("\n--- Analyse : Effet du Froid (<15°C) ---")
    results['cold'] = run_causal_inference(df_c, treatment='is_cold')
    
    return results

if __name__ == "__main__":
    from .preprocessing import run_pipeline
    df = run_pipeline()
    analyze_all_causal(df)
