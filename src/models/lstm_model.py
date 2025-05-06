#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implémentation du modèle LSTM pour la prédiction des tendances d'affluence hospitalière.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import pickle
from datetime import timedelta

class LSTMModel:
    """Classe pour l'implémentation du modèle LSTM."""
    
    def __init__(self, input_shape, output_shape=1, units=64, dropout_rate=0.2, 
                 learning_rate=0.001):
        """
        Initialisation du modèle LSTM.
        
        Arguments:
            input_shape (tuple): Forme des données d'entrée (timesteps, features)
            output_shape (int): Nombre de sorties (horizon de prédiction)
            units (int): Nombre d'unités LSTM
            dropout_rate (float): Taux de dropout
            learning_rate (float): Taux d'apprentissage
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Construction de l'architecture du modèle LSTM.
        
        Returns:
            self: Instance avec le modèle construit
        """
        print(f"Construction du modèle LSTM...")
        print(f"Forme d'entrée: {self.input_shape}")
        print(f"Forme de sortie: {self.output_shape}")
        
        # Création du modèle séquentiel
        model = Sequential()
        
        # Première couche LSTM
        model.add(LSTM(units=self.units, 
                      return_sequences=True, 
                      input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Deuxième couche LSTM
        model.add(LSTM(units=self.units // 2, 
                      return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Couche de sortie
        model.add(Dense(units=self.output_shape))
        
        # Compilation du modèle
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        print(self.model.summary())
        
        return self
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
            patience=20, verbose=1, save_path=None):
        """
        Entraîne le modèle LSTM sur les données d'entraînement.
        
        Arguments:
            X_train (numpy.array): Données d'entraînement
            y_train (numpy.array): Cibles d'entraînement
            X_val (numpy.array): Données de validation
            y_val (numpy.array): Cibles de validation
            epochs (int): Nombre d'époques
            batch_size (int): Taille des batchs
            patience (int): Patience pour l'early stopping
            verbose (int): Niveau de verbosité
            save_path (str): Chemin pour sauvegarder le modèle
            
        Returns:
            self: Instance avec le modèle entraîné
        """
        if self.model is None:
            self.build_model()
        
        # Configuration des callbacks
        callbacks = []
        
        # Early stopping pour éviter le surapprentissage
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Réduction du taux d'apprentissage
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Sauvegarde du meilleur modèle
        if save_path:
            # Création du répertoire si nécessaire
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            model_checkpoint = ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)
        
        # Entraînement du modèle
        print(f"Entraînement du modèle LSTM sur {len(X_train)} échantillons...")
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history.history
        
        return self
    
    def predict(self, X):
        """
        Prédit les valeurs futures.
        
        Arguments:
            X (numpy.array): Données d'entrée
            
        Returns:
            numpy.array: Prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit et entraîné avant de faire des prédictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur un ensemble de test.
        
        Arguments:
            X_test (numpy.array): Données de test
            y_test (numpy.array): Cibles de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit et entraîné avant d'être évalué")
        
        # Évaluation du modèle
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Prédiction sur l'ensemble de test
        y_pred = self.model.predict(X_test)
        
        # Calcul des métriques
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Calcul du MAPE (évite la division par zéro)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        metrics = {
            'Loss': loss,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        print(f"Évaluation du modèle sur {len(X_test)} échantillons:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_history(self, figsize=(12, 6)):
        """
        Affiche l'historique d'entraînement du modèle.
        
        Arguments:
            figsize (tuple): Taille de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure générée
        """
        if self.history is None:
            raise ValueError("Le modèle doit être entraîné avant d'afficher l'historique")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Affichage de la perte
        ax1.plot(self.history['loss'], label='Entraînement')
        if 'val_loss' in self.history:
            ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_title('Perte (Loss)')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('MSE')
        ax1.legend()
        
        # Affichage de l'erreur moyenne absolue
        ax2.plot(self.history['mae'], label='Entraînement')
        if 'val_mae' in self.history:
            ax2.plot(self.history['val_mae'], label='Validation')
        ax2.set_title('Erreur moyenne absolue (MAE)')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, X_test, y_test, scaler=None, target_idx=0, dates=None, figsize=(12, 6)):
        """
        Affiche les prédictions comparées aux valeurs réelles.
        
        Arguments:
            X_test (numpy.array): Données de test
            y_test (numpy.array): Cibles de test
            scaler (sklearn.preprocessing.MinMaxScaler): Scaler pour inverse_transform
            target_idx (int): Index de la colonne cible
            dates (pandas.DatetimeIndex): Dates pour l'axe x
            figsize (tuple): Taille de la figure
            
        Returns:
            matplotlib.figure.Figure: Figure générée
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit et entraîné avant d'afficher les prédictions")
        
        # Prédiction sur l'ensemble de test
        y_pred = self.model.predict(X_test)
        
        # Dénormalisation des données si un scaler est fourni
        if scaler is not None:
            from src.data.data_loader import inverse_transform_predictions
            y_test = inverse_transform_predictions(y_test, scaler, target_idx)
            y_pred = inverse_transform_predictions(y_pred, scaler, target_idx)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Utilisation des dates si fournies
        if dates is not None:
            ax.plot(dates, y_test, label='Valeurs réelles')
            ax.plot(dates, y_pred, label='Prédictions', color='red')
        else:
            ax.plot(y_test, label='Valeurs réelles')
            ax.plot(y_pred, label='Prédictions', color='red')
        
        ax.set_title('Prédictions LSTM vs Valeurs Réelles')
        ax.set_xlabel('Date' if dates is not None else 'Échantillon')
        ax.set_ylabel('Nombre de visites')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def forecast_future(self, last_sequence, steps=30, scaler=None, target_idx=0):
        """
        Prévoit les valeurs futures de manière récursive.
        
        Arguments:
            last_sequence (numpy.array): Dernière séquence d'entrée
            steps (int): Nombre de pas de temps à prévoir
            scaler (sklearn.preprocessing.MinMaxScaler): Scaler pour inverse_transform
            target_idx (int): Index de la colonne cible
            
        Returns:
            numpy.array: Prévisions futures
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit et entraîné avant de faire des prévisions")
        
        # Copie de la dernière séquence
        current_sequence = last_sequence.copy()
        
        # Liste pour stocker les prévisions
        forecasts = []
        
        # Prévision récursive
        for _ in range(steps):
            # Forme [1, timesteps, features]
            pred = self.model.predict(current_sequence.reshape(1, *current_sequence.shape))
            
            # Ajout de la prévision à la liste
            forecasts.append(pred[0])
            
            # Mise à jour de la séquence pour la prochaine prévision
            # On fait glisser la fenêtre d'une unité et on ajoute la prévision
            current_sequence = np.roll(current_sequence, -1, axis=0)
            
            # Pour une prévision multivariée, il faut mettre à jour uniquement la valeur cible
            if len(pred.shape) > 1 and pred.shape[1] > 1:
                current_sequence[-1, target_idx] = pred[0, 0]
            else:
                current_sequence[-1, target_idx] = pred[0]
        
        # Conversion en array
        forecasts = np.array(forecasts)
        
        # Dénormalisation des prévisions si un scaler est fourni
        if scaler is not None:
            from src.data.data_loader import inverse_transform_predictions
            forecasts = inverse_transform_predictions(forecasts, scaler, target_idx)
        
        return forecasts
    
    def save(self, model_path, metadata_path=None):
        """
        Sauvegarde le modèle entraîné.
        
        Arguments:
            model_path (str): Chemin pour sauvegarder le modèle Keras
            metadata_path (str): Chemin pour sauvegarder les métadonnées
        """
        if self.model is None:
            raise ValueError("Le modèle doit être construit avant d'être sauvegardé")
        
        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Sauvegarde du modèle
        self.model.save(model_path)
        print(f"Modèle sauvegardé dans {model_path}")
        
        # Sauvegarde des métadonnées
        if metadata_path:
            # Création du répertoire si nécessaire
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            
            metadata = {
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'units': self.units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'history': self.history
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"Métadonnées sauvegardées dans {metadata_path}")
    
    @classmethod
    def load(cls, model_path, metadata_path=None):
        """
        