#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fonctions pour charger et préparer les données pour les modèles prédictifs.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(file_path, parse_dates=True):
    """
    Charge les données depuis un fichier CSV.
    
    Arguments:
        file_path (str): Chemin du fichier à charger
        parse_dates (bool): Si True, parse la colonne 'date' en datetime
        
    Returns:
        pandas.DataFrame: Données chargées
    """
    if parse_dates:
        return pd.read_csv(file_path, parse_dates=['date'])
    else:
        return pd.read_csv(file_path)

def prepare_data_for_arima(df, target_column='hospital_visits'):
    """
    Prépare les données pour un modèle ARIMA.
    
    Arguments:
        df (pandas.DataFrame): Données d'entrée
        target_column (str): Colonne cible à prédire
        
    Returns:
        pandas.Series: Série temporelle pour ARIMA
    """
    # Pour ARIMA, on a besoin d'une série temporelle simple
    ts = df.set_index('date')[target_column]
    return ts

def prepare_data_for_lstm(df, target_column='hospital_visits', sequence_length=30, 
                          features=None, prediction_horizon=7):
    """
    Prépare les données pour un modèle LSTM.
    
    Arguments:
        df (pandas.DataFrame): Données d'entrée
        target_column (str): Colonne cible à prédire
        sequence_length (int): Nombre de pas de temps précédents à utiliser
        features (list): Liste des colonnes à utiliser comme features
        prediction_horizon (int): Horizon de prédiction (nombre de jours à prédire)
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    print(f"Préparation des données pour LSTM...")
    
    # Vérification des données d'entrée
    if 'date' not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'date'")
    
    # Si aucune feature n'est spécifiée, on utilise toutes les colonnes numériques
    if features is None:
        features = [col for col in df.columns if col not in ['date'] and df[col].dtype != 'object']
    
    # Séparation des données en ensemble d'apprentissage et de test (chronologiquement)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Taille de l'ensemble d'apprentissage: {len(train_df)}")
    print(f"Taille de l'ensemble de test: {len(test_df)}")
    
    # Normalisation des données
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])
    
    # Création des séquences
    X_train, y_train = create_sequences(train_scaled, sequence_length, 
                                      features.index(target_column), prediction_horizon)
    X_test, y_test = create_sequences(test_scaled, sequence_length, 
                                    features.index(target_column), prediction_horizon)
    
    print(f"Forme des données d'apprentissage: {X_train.shape}, {y_train.shape}")
    print(f"Forme des données de test: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, scaler, features

def create_sequences(data, sequence_length, target_idx, prediction_horizon=1):
    """
    Crée des séquences pour l'apprentissage du LSTM.
    
    Arguments:
        data (numpy.array): Données normalisées
        sequence_length (int): Longueur de la séquence d'entrée
        target_idx (int): Index de la colonne cible dans les données
        prediction_horizon (int): Nombre de pas de temps à prédire
        
    Returns:
        tuple: (X, y) où X est l'entrée et y est la sortie
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i:i+sequence_length])
        
        # Pour prédire plusieurs pas de temps
        if prediction_horizon > 1:
            y.append(data[i+sequence_length:i+sequence_length+prediction_horizon, target_idx])
        else:
            y.append(data[i+sequence_length, target_idx])
    
    return np.array(X), np.array(y)

def inverse_transform_predictions(predictions, scaler, target_column_idx):
    """
    Inverse la normalisation des prédictions.
    
    Arguments:
        predictions (numpy.array): Prédictions normalisées
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler utilisé pour la normalisation
        target_column_idx (int): Index de la colonne cible
        
    Returns:
        numpy.array: Prédictions dénormalisées
    """
    # Création d'un array de zéros avec la même forme que les données d'origine
    dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
    
    # Si les prédictions sont multidimensionnelles (plusieurs pas de temps)
    if len(predictions.shape) > 1:
        # Pour chaque pas de temps
        result = []
        for i in range(predictions.shape[1]):
            dummy_i = dummy.copy()
            dummy_i[:, target_column_idx] = predictions[:, i]
            result.append(scaler.inverse_transform(dummy_i)[:, target_column_idx])
        return np.array(result).T
    else:
        # Pour une prédiction à un seul pas de temps
        dummy[:, target_column_idx] = predictions
        return scaler.inverse_transform(dummy)[:, target_column_idx]

def get_service_data(df, service_name):
    """
    Récupère les données pour un service spécifique.
    
    Arguments:
        df (pandas.DataFrame): Données d'entrée
        service_name (str): Nom du service
        
    Returns:
        pandas.Series: Série temporelle pour le service spécifié
    """
    if service_name not in df.columns:
        raise ValueError(f"Le service '{service_name}' n'existe pas dans les données")
    
    return df.set_index('date')[service_name]

def get_last_n_days(df, n=30):
    """
    Récupère les derniers N jours de données.
    
    Arguments:
        df (pandas.DataFrame): Données d'entrée
        n (int): Nombre de jours à récupérer
        
    Returns:
        pandas.DataFrame: Données des N derniers jours
    """
    return df.sort_values('date').tail(n)

if __name__ == '__main__':
    # Test des fonctions
    df = load_data('data/processed/hospital_data_processed.csv')
    print(f"Données chargées avec succès. Dimensions: {df.shape}")
    print(f"Colonnes disponibles: {df.columns.tolist()}")
    
    # Test de la préparation pour ARIMA
    ts = prepare_data_for_arima(df)
    print(f"Série temporelle pour ARIMA: {ts.head()}")
    
    # Test de la préparation pour LSTM
    X_train, y_train, X_test, y_test, scaler, features = prepare_data_for_lstm(
        df, features=['hospital_visits', 'visits_lag_1', 'visits_lag_7', 'visits_ma_7', 
                      'is_weekend', 'is_holiday', 'month']
    )
    print(f"Préparation LSTM réussie.")
