#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'entraînement pour les modèles prédictifs de tendances hospitalières.
Permet d'entraîner et d'évaluer les modèles ARIMA et LSTM.
"""

import os
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import logging

# Imports pour l'évaluation des modèles
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Import des modules personnalisés
from src.models.arima_model import ArimaModel
from src.models.lstm_model import LSTMModel

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Charge les données prétraitées.
    
    Args:
        filepath: Chemin vers le fichier de données
        
    Returns:
        DataFrame pandas contenant les données
    """
    logger.info(f"Chargement des données depuis {filepath}")
    df = pd.read_csv(filepath)
    
    # Conversion de la colonne date en datetime si elle existe
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def prepare_time_series_data(df, target_col, date_col='date', test_size=0.2):
    """
    Prépare les données pour les modèles de séries temporelles.
    
    Args:
        df: DataFrame pandas avec les données
        target_col: Nom de la colonne cible à prédire
        date_col: Nom de la colonne contenant les dates
        test_size: Proportion des données à utiliser pour le test
        
    Returns:
        Tuple (train_data, test_data, X_train, y_train, X_test, y_test)
    """
    logger.info("Préparation des données pour les modèles de séries temporelles")
    
    # Tri des données par date
    df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Division en ensembles d'entraînement et de test (temporelle)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_data = df_sorted.iloc[:split_idx].copy()
    test_data = df_sorted.iloc[split_idx:].copy()
    
    logger.info(f"Données divisées en {len(train_data)} échantillons d'entraînement et {len(test_data)} échantillons de test")
    
    # Préparation des caractéristiques et des cibles pour les modèles
    if date_col in df.columns:
        X_train = train_data.drop(columns=[target_col, date_col])
        X_test = test_data.drop(columns=[target_col, date_col])
    else:
        X_train = train_data.drop(columns=[target_col])
        X_test = test_data.drop(columns=[target_col])
    
    y_train = train_data[target_col]
    y_test = test_data[target_col]
    
    return train_data, test_data, X_train, y_train, X_test, y_test

def train_arima_model(train_data, target_col, date_col='date'):
    """
    Entraîne un modèle ARIMA.
    
    Args:
        train_data: DataFrame pandas contenant les données d'entraînement
        target_col: Nom de la colonne cible à prédire
        date_col: Nom de la colonne contenant les dates
        
    Returns:
        Modèle ARIMA entraîné
    """
    logger.info("Entraînement du modèle ARIMA")
    
    # Préparation des données pour ARIMA
    train_series = train_data.set_index(date_col)[target_col]
    
    # Création et entraînement du modèle ARIMA
    arima_model = ArimaModel()
    arima_model.fit(train_series)
    
    logger.info("Modèle ARIMA entraîné avec succès")
    
    return arima_model

def train_lstm_model(X_train, y_train, X_test, y_test, window_size=30, forecast_horizon=7):
    """
    Prépare les données et entraîne un modèle LSTM.
    
    Args:
        X_train: Caractéristiques d'entraînement
        y_train: Cibles d'entraînement
        X_test: Caractéristiques de test
        y_test: Cibles de test
        window_size: Taille de la fenêtre d'historique
        forecast_horizon: Horizon de prévision
        
    Returns:
        Modèle LSTM entraîné
    """
    logger.info("Préparation des données pour le modèle LSTM")
    
    # Création du modèle LSTM
    lstm_model = LSTMModel(
        input_shape=(window_size, X_train.shape[1]),
        output_dim=forecast_horizon
    )
    
    # Préparation des séquences pour LSTM
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, window_size, forecast_horizon)
    X_test_seq, y_test_seq = prepare_sequences(X_test, y_test, window_size, forecast_horizon)
    
    # Entraînement du modèle LSTM
    logger.info("Entraînement du modèle LSTM")
    lstm_model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq))
    
    logger.info("Modèle LSTM entraîné avec succès")
    
    return lstm_model, (X_train_seq, y_train_seq, X_test_seq, y_test_seq)

def prepare_sequences(X, y, window_size, forecast_horizon):
    """
    Prépare les séquences pour le modèle LSTM.
    
    Args:
        X: Caractéristiques
        y: Cibles
        window_size: Taille de la fenêtre d'historique
        forecast_horizon: Horizon de prévision
        
    Returns:
        Tuple (X_seq, y_seq) pour l'entraînement LSTM
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - window_size - forecast_horizon + 1):
        X_seq.append(X.iloc[i:i+window_size].values)
        y_seq.append(y.iloc[i+window_size:i+window_size+forecast_horizon].values)
    
    return np.array(X_seq), np.array(y_seq)

def evaluate_model(model, X_test, y_test, model_type='arima'):
    """
    Évalue les performances d'un modèle.
    
    Args:
        model: Modèle entraîné (ARIMA ou LSTM)
        X_test: Caractéristiques de test
        y_test: Cibles de test
        model_type: Type de modèle ('arima' ou 'lstm')
        
    Returns:
        Dictionnaire contenant les métriques d'évaluation
    """
    logger.info(f"Évaluation du modèle {model_type}")
    
    if model_type == 'arima':
        # Prédiction avec le modèle ARIMA
        y_pred = model.predict(len(y_test))
    else:  # lstm
        # Prédiction avec le modèle LSTM
        y_pred = model.predict(X_test)
        
        # Reshape des prédictions si nécessaire
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = y_pred[:, 0]  # Prendre la première valeur de chaque séquence prédite
    
    # Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Affichage des métriques
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test
    }

def plot_predictions(y_test, y_pred, model_type, output_dir=None):
    """
    Trace les prédictions par rapport aux valeurs réelles.
    
    Args:
        y_test: Valeurs réelles
        y_pred: Valeurs prédites
        model_type: Type de modèle ('arima' ou 'lstm')
        output_dir: Répertoire où sauvegarder le graphique
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Valeurs réelles')
    plt.plot(y_pred, label='Prédictions', linestyle='--')
    plt.title(f'Prédictions vs Valeurs réelles - Modèle {model_type.upper()}')
    plt.xlabel('Temps')
    plt.ylabel('Nombre de patients')
    plt.legend()
    plt.grid(True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{model_type}_predictions.png'))
    
    plt.close()

def save_model(model, model_type, output_dir='../../models'):
    """
    Sauvegarde le modèle entraîné.
    
    Args:
        model: Modèle entraîné
        model_type: Type de modèle ('arima' ou 'lstm')
        output_dir: Répertoire où sauvegarder le modèle
    """
    # Création du répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Génération du nom de fichier avec horodatage
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_type}_model_{timestamp}"
    
    if model_type == 'arima':
        # Sauvegarde du modèle ARIMA avec pickle
        with open(os.path.join(output_dir, f"{filename}.pkl"), 'wb') as f:
            pickle.dump(model, f)
    else:  # lstm
        # Sauvegarde du modèle LSTM avec Keras
        model.save_model(os.path.join(output_dir, filename))
    
    logger.info(f"Modèle {model_type} sauvegardé dans {output_dir}")
    
    return os.path.join(output_dir, filename)

def main(data_filepath='../../data/processed/features_hospital_data.csv',
         target_col='patient_count',
         date_col='date',
         output_dir='../../models',
         plots_dir='../../reports/figures'):
    """
    Point d'entrée principal pour l'entraînement des modèles.
    
    Args:
        data_filepath: Chemin vers le fichier de données prétraitées
        target_col: Nom de la colonne cible à prédire
        date_col: Nom de la colonne contenant les dates
        output_dir: Répertoire où sauvegarder les modèles
        plots_dir: Répertoire où sauvegarder les graphiques
    """
    # Chargement des données
    df = load_data(data_filepath)
    
    # Préparation des données pour les modèles
    train_data, test_data, X_train, y_train, X_test, y_test = prepare_time_series_data(
        df, target_col, date_col
    )
    
    # Entraînement et évaluation du modèle ARIMA
    arima_model = train_arima_model(train_data, target_col, date_col)
    arima_results = evaluate_model(arima_model, None, test_data[target_col], 'arima')
    plot_predictions(
        arima_results['y_test'], 
        arima_results['y_pred'], 
        'arima', 
        plots_dir
    )
    save_model(arima_model, 'arima', output_dir)
    
    # Entraînement et évaluation du modèle LSTM
    window_size = 30
    forecast_horizon = 7
    lstm_model, lstm_data = train_lstm_model(
        X_train, y_train, X_test, y_test,
        window_size, forecast_horizon
    )
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = lstm_data
    lstm_results = evaluate_model(lstm_model, X_test_seq, y_test_seq, 'lstm')
    plot_predictions(
        lstm_results['y_test'][:, 0],  # Premier jour de prévision
        lstm_results['y_pred'][:, 0],  # Premier jour de prévision
        'lstm',
        plots_dir
    )
    save_model(lstm_model, 'lstm', output_dir)
    
    logger.info("Entraînement et évaluation des modèles terminés avec succès")

if __name__ == '__main__':
    main()