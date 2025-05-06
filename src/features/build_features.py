#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la création et transformation de caractéristiques (features)
pour alimenter les modèles prédictifs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_lag_features(df, target_col, lag_periods=[1, 7, 14, 30]):
    """
    Crée des variables décalées (lag features) pour le modèle prédictif.
    
    Args:
        df: DataFrame pandas avec une colonne de date triée chronologiquement
        target_col: Nom de la colonne cible pour laquelle créer des lags
        lag_periods: Liste des périodes de décalage à créer
        
    Returns:
        DataFrame pandas avec les caractéristiques de lag ajoutées
    """
    logger.info(f"Création de caractéristiques décalées pour {target_col}")
    
    df_copy = df.copy()
    
    for lag in lag_periods:
        lag_col_name = f'{target_col}_lag_{lag}'
        df_copy[lag_col_name] = df_copy[target_col].shift(lag)
        logger.info(f"Ajout de la caractéristique {lag_col_name}")
    
    # Suppression des lignes avec valeurs NaN résultant des lags
    df_copy = df_copy.dropna()
    logger.info(f"Suppression de {len(df) - len(df_copy)} lignes avec valeurs manquantes après création des lags")
    
    return df_copy

def create_time_based_features(df, date_col='date'):
    """
    Extraction de caractéristiques basées sur la date/heure.
    
    Args:
        df: DataFrame pandas avec une colonne de date
        date_col: Nom de la colonne contenant les dates
        
    Returns:
        DataFrame pandas avec les caractéristiques temporelles
    """
    logger.info("Création de caractéristiques temporelles avancées")
    
    df_copy = df.copy()
    
    # Vérification que la colonne date est bien au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Caractéristiques cycliques pour représenter la saisonnalité
    
    # Jour de la semaine (caractéristiques cycliques)
    df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy[date_col].dt.dayofweek / 7)
    df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy[date_col].dt.dayofweek / 7)
    
    # Mois de l'année (caractéristiques cycliques)
    df_copy['month_sin'] = np.sin(2 * np.pi * (df_copy[date_col].dt.month - 1) / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * (df_copy[date_col].dt.month - 1) / 12)
    
    # Jour du mois
    df_copy['day_sin'] = np.sin(2 * np.pi * (df_copy[date_col].dt.day - 1) / 31)
    df_copy['day_cos'] = np.cos(2 * np.pi * (df_copy[date_col].dt.day - 1) / 31)
    
    # Caractéristiques de tendance
    df_copy['days_from_start'] = (df_copy[date_col] - df_copy[date_col].min()).dt.days
    
    logger.info("Caractéristiques temporelles créées avec succès")
    
    return df_copy

def scale_features(df, target_col, categorical_cols=None, scaler_type='standard'):
    """
    Normalise les caractéristiques numériques et encode les caractéristiques catégorielles.
    
    Args:
        df: DataFrame pandas avec les caractéristiques
        target_col: Nom de la colonne cible à ne pas transformer
        categorical_cols: Liste des colonnes catégorielles à encoder
        scaler_type: Type de scaling ('standard' ou 'minmax')
        
    Returns:
        Tuple (DataFrame pandas avec caractéristiques transformées, objet transformer)
    """
    logger.info(f"Scaling des caractéristiques avec méthode {scaler_type}")
    
    df_copy = df.copy()
    
    if categorical_cols is None:
        categorical_cols = []
    
    # Identification des colonnes numériques à transformer
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclusion de la colonne cible et des colonnes catégorielles
    numeric_cols = [col for col in numeric_cols if col != target_col and col not in categorical_cols]
    
    transformers = []
    
    # Configuration du scaler pour les colonnes numériques
    if scaler_type == 'standard':
        numeric_transformer = StandardScaler()
    else:  # minmax
        numeric_transformer = MinMaxScaler()
    
    transformers.append(('num', numeric_transformer, numeric_cols))
    
    # Configuration de l'encodeur pour les colonnes catégorielles
    if categorical_cols:
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformers.append(('cat', cat_transformer, categorical_cols))
    
    # Création et application du transformateur
    preprocessor = ColumnTransformer(transformers)
    
    # Séparation des caractéristiques et de la cible
    X = df_copy.drop(columns=[target_col])
    y = df_copy[target_col]
    
    # Application des transformations
    X_transformed = pd.DataFrame(
        preprocessor.fit_transform(X),
        columns=get_feature_names(preprocessor, X.columns)
    )
    
    # Réassemblage du DataFrame
    df_transformed = pd.concat([X_transformed, y.reset_index(drop=True)], axis=1)
    
    logger.info("Scaling des caractéristiques terminé")
    
    return df_transformed, preprocessor

def get_feature_names(column_transformer, original_feature_names):
    """
    Récupère les noms des caractéristiques après transformation.
    
    Args:
        column_transformer: Objet ColumnTransformer qui a été ajusté
        original_feature_names: Noms des caractéristiques d'origine
        
    Returns:
        Liste des noms de caractéristiques après transformation
    """
    feature_names = []
    
    for name, transformer, features in column_transformer.transformers_:
        if name == 'drop' or transformer == 'drop':
            continue
            
        if hasattr(transformer, 'get_feature_names_out'):
            # Pour les transformateurs comme OneHotEncoder
            transformed_names = transformer.get_feature_names_out(features)
        else:
            # Pour les transformateurs comme StandardScaler
            transformed_names = features
            
        feature_names.extend(transformed_names)
        
    return feature_names

def prepare_features_for_time_series(df, target_col, date_col='date', window_size=30, forecast_horizon=7):
    """
    Prépare les données pour les modèles de séries temporelles (comme LSTM).
    
    Args:
        df: DataFrame pandas avec les données
        target_col: Nom de la colonne cible à prédire
        date_col: Nom de la colonne contenant les dates
        window_size: Taille de la fenêtre d'historique (jours)
        forecast_horizon: Horizon de prévision (jours)
        
    Returns:
        Tuple (X, y) pour l'entraînement du modèle
    """
    logger.info(f"Préparation des caractéristiques pour séries temporelles (window={window_size}, horizon={forecast_horizon})")
    
    # Tri des données par date
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(df_sorted) - window_size - forecast_horizon + 1):
        # Séquence d'entrée (window_size jours d'historique)
        X_seq = df_sorted.iloc[i:i+window_size][target_col].values
        
        # Séquence cible (forecast_horizon jours à prédire)
        y_seq = df_sorted.iloc[i+window_size:i+window_size+forecast_horizon][target_col].values
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    # Conversion en tableaux numpy
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    # Reshape pour les modèles LSTM: [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    logger.info(f"Données préparées: X shape={X.shape}, y shape={y.shape}")
    
    return X, y

def main(input_filepath='../../data/processed/processed_hospital_data.csv',
         output_filepath='../../data/processed/features_hospital_data.csv',
         target_column='patient_count'):
    """
    Point d'entrée principal pour la création des caractéristiques.
    
    Args:
        input_filepath: Chemin vers le fichier de données prétraitées
        output_filepath: Chemin où sauvegarder les données avec caractéristiques
        target_column: Nom de la colonne cible pour les prédictions
    """
    # Chargement des données prétraitées
    logger.info(f"Chargement des données prétraitées depuis {input_filepath}")
    df = pd.read_csv(input_filepath)
    
    # Conversion de la colonne date en datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Création de caractéristiques temporelles avancées
    df = create_time_based_features(df)
    
    # Création de caractéristiques de lag
    df = create_lag_features(df, target_column)
    
    # Scaling des caractéristiques
    categorical_cols = ['is_weekend', 'is_holiday']
    df_scaled, _ = scale_features(df, target_column, categorical_cols)
    
    # Sauvegarde des données avec caractéristiques
    logger.info(f"Sauvegarde des données avec caractéristiques vers {output_filepath}")
    df_scaled.to_csv(output_filepath, index=False)
    logger.info("Création de caractéristiques terminée avec succès")

if __name__ == '__main__':
    main()
