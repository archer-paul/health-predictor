#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implémentation du modèle ARIMA pour la prédiction des tendances d'affluence hospitalière.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pickle
import warnings
from datetime import timedelta

class ARIMAModel:
    """Classe pour l'implémentation du modèle ARIMA."""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Initialisation du modèle ARIMA.
        
        Arguments:
            order (tuple): Ordre du modèle ARIMA (p, d, q)
            seasonal_order (tuple): Ordre saisonnier du modèle SARIMAX (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
        self.is_seasonal = seasonal_order is not None
    
    def check_stationarity(self, ts):
        """
        Vérifie la stationnarité de la série temporelle avec le test Augmented Dickey-Fuller.
        
        Arguments:
            ts (pandas.Series): Série temporelle à analyser
            
        Returns:
            bool: True si la série est stationnaire, False sinon
        """
        result = adfuller(ts.dropna())
        print(f'Test ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        
        # Seuil de signification de 5%
        if result[1] <= 0.05:
            print("La série est stationnaire (rejet de l'hypothèse nulle)")
            return True
        else:
            print("La série n'est pas stationnaire (non-rejet de l'hypothèse nulle)")
            return False
    
    def plot_diagnostics(self, ts):
        """
        Affiche les diagnostics de la série temporelle (ACF, PACF).
        
        Arguments:
            ts (pandas.Series): Série temporelle à analyser
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF et PACF
        plot_acf(ts.dropna(), ax=ax1)
        plot_pacf(ts.dropna(), ax=ax2)
        
        ax1.set_title('Autocorrelation Function (ACF)')
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        return fig
    
    def find_best_parameters(self, ts, p_range=(0, 2), d_range=(0, 2), q_range=(0, 2)):
        """
        Recherche les meilleurs paramètres pour le modèle ARIMA.
        
        Arguments:
            ts (pandas.Series): Série temporelle à analyser
            p_range (tuple): Plage de valeurs pour p
            d_range (tuple): Plage de valeurs pour d
            q_range (tuple): Plage de valeurs pour q
            
        Returns:
            tuple: Meilleurs paramètres (p, d, q)
        """
        best_aic = float('inf')
        best_order = None
        
        # Suppression des avertissements pour la recherche de paramètres
        warnings.filterwarnings('ignore')
        
        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        results = model.fit()
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                            print(f"Nouveau meilleur ordre trouvé: {best_order} avec AIC: {best_aic}")
                    except:
                        continue
        
        # Réactivation des avertissements
        warnings.resetwarnings()
        
        print(f"Meilleur ordre ARIMA: {best_order} avec AIC: {best_aic}")
        return best_order
    
    def fit(self, ts):
        """
        Entraîne le modèle ARIMA sur la série temporelle.
        
        Arguments:
            ts (pandas.Series): Série temporelle d'entraînement
            
        Returns:
            self: Instance du modèle entraîné
        """
        print(f"Entraînement du modèle ARIMA avec ordre {self.order}")
        
        if self.is_seasonal:
            print(f"Utilisation de SARIMAX avec ordre saisonnier {self.seasonal_order}")
            self.model = SARIMAX(ts, order=self.order, seasonal_order=self.seasonal_order)
        else:
            self.model = ARIMA(ts, order=self.order)
        
        self.results = self.model.fit()
        print(f"Modèle entraîné avec succès. AIC: {self.results.aic}")
        
        return self
    
    def predict(self, steps=30):
        """
        Prédit les valeurs futures.
        
        Arguments:
            steps (int): Nombre de pas de temps à prédire
            
        Returns:
            pandas.Series: Prédictions
        """
        if self.results is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        # Récupération de la dernière date de la série temporelle
        last_date = self.results.model.data.dates[-1]
        
        # Création de l'index pour les prédictions
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
        
        # Prédiction
        forecast = self.results.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        
        # Intervalles de confiance
        conf_int = forecast.conf_int()
        
        # Attribution de l'index aux prédictions
        forecast_mean.index = forecast_index
        conf_int.index = forecast_index
        
        return forecast_mean, conf_int
    
    def evaluate(self, ts_test):
        """
        Évalue le modèle sur un ensemble de test.
        
        Arguments:
            ts_test (pandas.Series): Série temporelle de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        if self.results is None:
            raise ValueError("Le modèle doit être entraîné avant d'être évalué")
        
        # Prédiction sur l'ensemble de test
        predictions = self.results.get_prediction(start=ts_test.index[0], end=ts_test.index[-1])
        pred_mean = predictions.predicted_mean
        
        # Calcul des métriques
        mse = np.mean((ts_test - pred_mean) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(ts_test - pred_mean))
        mape = np.mean(np.abs((ts_test - pred_mean) / ts_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        print(f"Évaluation du modèle sur {len(ts_test)} points de données:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_predictions(self, ts, predictions, conf_int=None, figsize=(12, 6)):
        """
        Affiche les prédictions comparées aux valeurs réelles.
        
        Arguments:
            ts (pandas.Series): Série temporelle réelle
            predictions (pandas.Series): Prédictions
            conf_int (pandas.DataFrame): Intervalles de confiance
            figsize (tuple): Taille de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure générée
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Affichage des valeurs réelles
        ts.plot(ax=ax, label='Valeurs réelles')
        
        # Affichage des prédictions
        predictions.plot(ax=ax, label='Prédictions', color='red')
        
        # Affichage des intervalles de confiance
        if conf_int is not None:
            ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                           color='pink', alpha=0.3, label='Intervalle de confiance (95%)')
        
        ax.set_title('Prédictions ARIMA vs Valeurs Réelles')
        ax.set_xlabel('Date')
        ax.set_ylabel('Nombre de visites')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def save(self, file_path):
        """
        Sauvegarde le modèle entraîné.
        
        Arguments:
            file_path (str): Chemin de sauvegarde
        """
        if self.results is None:
            raise ValueError("Le modèle doit être entraîné avant d'être sauvegardé")
        
        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Sauvegarde du modèle
        with open(file_path, 'wb') as f:
            pickle.dump({
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'is_seasonal': self.is_seasonal,
                'results': self.results
            }, f)
        
        print(f"Modèle sauvegardé dans {file_path}")
    
    @classmethod
    def load(cls, file_path):
        """
        Charge un modèle entraîné.
        
        Arguments:
            file_path (str): Chemin du modèle
            
        Returns:
            ARIMAModel: Instance du modèle chargé
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Création d'une nouvelle instance
        model = cls(order=model_data['order'], seasonal_order=model_data['seasonal_order'])
        model.is_seasonal = model_data['is_seasonal']
        model.results = model_data['results']
        
        return model

def run_arima_analysis(ts, train_test_split=0.8, order=None, seasonal_order=None, 
                      forecast_steps=30, save_path=None):
    """
    Exécute une analyse ARIMA complète sur une série temporelle.
    
    Arguments:
        ts (pandas.Series): Série temporelle à analyser
        train_test_split (float): Proportion de données pour l'entraînement
        order (tuple): Ordre du modèle ARIMA (p, d, q)
        seasonal_order (tuple): Ordre saisonnier du modèle SARIMA (P, D, Q, s)
        forecast_steps (int): Nombre de pas de temps à prédire dans le futur
        save_path (str): Chemin pour sauvegarder le modèle
        
    Returns:
        tuple: (modèle, métriques, prédictions, intervalles de confiance)
    """
    # Division des données
    train_size = int(len(ts) * train_test_split)
    train, test = ts[:train_size], ts[train_size:]
    
    print(f"Données d'entraînement: {len(train)} observations")
    print(f"Données de test: {len(test)} observations")
    
    # Création du modèle
    model = ARIMAModel(order=order if order else (1, 1, 1), 
                      seasonal_order=seasonal_order)
    
    # Vérification de la stationnarité
    model.check_stationarity(train)
    
    # Recherche des meilleurs paramètres si non spécifiés
    if order is None:
        best_order = model.find_best_parameters(train)
        model.order = best_order
    
    # Entraînement du modèle
    model.fit(train)
    
    # Évaluation sur l'ensemble de test
    metrics = model.evaluate(test)
    
    # Prédiction future
    forecast, conf_int = model.predict(steps=forecast_steps)
    
    # Sauvegarde du modèle si spécifié
    if save_path:
        model.save(save_path)
    
    return model, metrics, forecast, conf_int

if __name__ == '__main__':
    # Exemple d'utilisation
    from src.data.data_loader import load_data, prepare_data_for_arima
    
    # Chargement des données
    df = load_data('data/processed/hospital_data_processed.csv')
    ts = prepare_data_for_arima(df)
    
    # Analyse ARIMA
    model, metrics, forecast, conf_int = run_arima_analysis(
        ts, 
        train_test_split=0.8,
        order=(2, 1, 2),  # Exemple d'ordre ARIMA
        seasonal_order=(1, 1, 1, 12),  # Exemple d'ordre saisonnier
        forecast_steps=30,
        save_path='models/arima_model.pkl'
    )
    
    # Affichage des prédictions
    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Observations')
    plt.plot(forecast, label='Prédictions', color='red')
    plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                    color='pink', alpha=0.3)
    plt.title('Prédictions ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Nombre de visites')
    plt.legend()
    plt.savefig('notebooks/arima_forecast.png')
    plt.close()
    
    print("Analyse ARIMA terminée.")
