#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour générer et prétraiter des données synthétiques d'affluence hospitalière.
Les données générées simulent des patterns saisonniers et hebdomadaires.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import argparse

def create_directory_if_not_exists(directory):
    """Crée un répertoire s'il n'existe pas déjà."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_synthetic_data(start_date='2020-01-01', end_date='2023-12-31'):
    """
    Génère des données synthétiques d'affluence hospitalière.
    
    Arguments:
        start_date (str): Date de début (format YYYY-MM-DD)
        end_date (str): Date de fin (format YYYY-MM-DD)
        
    Returns:
        pandas.DataFrame: Données générées
    """
    print(f"Génération de données synthétiques du {start_date} au {end_date}...")
    
    # Création d'une série de dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Création du dataframe
    df = pd.DataFrame({
        'date': date_range,
    })
    
    # Extraction des composantes temporelles
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day'] = df['date'].dt.day
    
    # Génération de la tendance de base
    base_visits = 200  # Nombre moyen de visites quotidiennes
    
    # Composante annuelle: plus de visites en hiver (grippes, rhumes, etc.)
    df['seasonal_yearly'] = 100 * np.sin(2 * np.pi * (df['month'] - 1) / 12)
    
    # Composante hebdomadaire: plus de visites les lundis et vendredis
    weekday_effect = {
        0: 50,    # Lundi
        1: 0,     # Mardi
        2: -20,   # Mercredi
        3: -10,   # Jeudi
        4: 40,    # Vendredi
        5: -30,   # Samedi
        6: -40    # Dimanche
    }
    df['seasonal_weekly'] = df['day_of_week'].map(weekday_effect)
    
    # Tendance à long terme (légère augmentation au fil des années)
    days_since_start = (df['date'] - df['date'].min()).dt.days
    df['trend'] = days_since_start * 0.1
    
    # Ajout d'événements spéciaux (épidémies, canicules, etc.)
    df['special_events'] = 0
    
    # Simulation d'épidémies hivernales
    for year in df['year'].unique():
        # Épidémie de grippe (décembre-février)
        winter_start = datetime(year-1, 12, 1) if year > int(start_date[:4]) else datetime(year, 12, 1)
        winter_end = datetime(year, 2, 28)
        winter_peak = winter_start + timedelta(days=random.randint(20, 60))
        
        for idx, row in df.iterrows():
            if winter_start <= row['date'] <= winter_end:
                days_from_peak = abs((row['date'] - winter_peak).total_seconds() / (24 * 3600))
                df.at[idx, 'special_events'] += max(0, 200 * np.exp(-0.001 * days_from_peak**2))
        
        # Canicule estivale (effet sur les visites)
        if random.random() > 0.3:  # 70% de chance d'avoir une canicule
            summer_start = datetime(year, 7, 1)
            summer_end = datetime(year, 8, 31)
            summer_peak = summer_start + timedelta(days=random.randint(10, 50))
            
            for idx, row in df.iterrows():
                if summer_start <= row['date'] <= summer_end:
                    days_from_peak = abs((row['date'] - summer_peak).total_seconds() / (24 * 3600))
                    df.at[idx, 'special_events'] += max(0, 100 * np.exp(-0.002 * days_from_peak**2))
    
    # Calcul du nombre total de visites
    df['hospital_visits'] = (
        base_visits 
        + df['seasonal_yearly'] 
        + df['seasonal_weekly'] 
        + df['trend'] 
        + df['special_events'] 
        + np.random.normal(0, 30, len(df))  # Bruit aléatoire
    )
    
    # Arrondir et s'assurer que les valeurs sont positives
    df['hospital_visits'] = np.maximum(0, np.round(df['hospital_visits'])).astype(int)
    
    # Services médicaux (répartition des visites par service)
    services = ['Urgences', 'Cardiologie', 'Pédiatrie', 'Orthopédie', 'Pneumologie']
    
    # Répartition par service
    base_distributions = {
        'Urgences': 0.35,
        'Cardiologie': 0.15,
        'Pédiatrie': 0.20,
        'Orthopédie': 0.18,
        'Pneumologie': 0.12
    }
    
    # Ajustements saisonniers par service
    for service in services:
        df[service] = df['hospital_visits'] * base_distributions[service]
        
        # Ajustements spécifiques par service
        if service == 'Pneumologie':
            # Plus de cas en hiver
            df[service] *= 1 + 0.5 * np.sin(2 * np.pi * (df['month'] - 1) / 12)
        elif service == 'Orthopédie':
            # Plus de cas en été (activités extérieures)
            df[service] *= 1 - 0.3 * np.sin(2 * np.pi * (df['month'] - 1) / 12)
        elif service == 'Pédiatrie':
            # Pics pendant les périodes scolaires
            school_effect = np.sin(2 * np.pi * (df['month'] - 9) / 12)
            df[service] *= 1 + 0.2 * school_effect
            
        # Arrondir
        df[service] = np.round(df[service]).astype(int)
    
    # Sélection des colonnes finales
    final_columns = ['date', 'hospital_visits'] + services
    
    return df[final_columns]

def save_data(df, output_path):
    """
    Sauvegarde les données générées.
    
    Arguments:
        df (pandas.DataFrame): Données à sauvegarder
        output_path (str): Chemin de sauvegarde
    """
    create_directory_if_not_exists(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    print(f"Données sauvegardées dans {output_path}")

def preprocess_data(input_path, output_path):
    """
    Prétraite les données pour l'analyse.
    
    Arguments:
        input_path (str): Chemin des données brutes
        output_path (str): Chemin de sauvegarde des données prétraitées
    """
    print(f"Prétraitement des données depuis {input_path}...")
    
    # Chargement des données
    df = pd.read_csv(input_path, parse_dates=['date'])
    
    # Ajout de variables temporelles utiles pour les modèles
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    # Création d'indicateurs pour les jours fériés (simplification)
    holidays = [
        # Jours fériés français (approximatifs)
        "01-01",  # Jour de l'an
        "05-01",  # Fête du travail
        "05-08",  # Victoire 1945
        "07-14",  # Fête nationale
        "08-15",  # Assomption
        "11-01",  # Toussaint
        "11-11",  # Armistice
        "12-25",  # Noël
    ]
    
    df['is_holiday'] = df['date'].dt.strftime('%m-%d').isin(holidays).astype(int)
    
    # Ajout de lag features (pour les modèles de séries temporelles)
    for lag in [1, 7, 14, 28]:
        df[f'visits_lag_{lag}'] = df['hospital_visits'].shift(lag)
    
    # Ajout de moving averages
    for window in [7, 14, 30]:
        df[f'visits_ma_{window}'] = df['hospital_visits'].rolling(window=window).mean()
    
    # Création d'une colonne pour chaque jour de la semaine (one-hot encoding)
    for i in range(7):
        df[f'dow_{i}'] = (df['dayofweek'] == i).astype(int)
    
    # Dropping NaN values (due to lags and moving averages)
    df = df.dropna()
    
    # Sauvegarde des données prétraitées
    create_directory_if_not_exists(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    print(f"Données prétraitées sauvegardées dans {output_path}")

def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description='Génération et prétraitement de données hospitalières')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Date de début (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='Date de fin (YYYY-MM-DD)')
    parser.add_argument('--raw-output', type=str, default='data/raw/hospital_data.csv', help='Chemin de sortie pour les données brutes')
    parser.add_argument('--processed-output', type=str, default='data/processed/hospital_data_processed.csv', help='Chemin de sortie pour les données prétraitées')
    args = parser.parse_args()
    
    # Génération des données synthétiques
    df = generate_synthetic_data(args.start_date, args.end_date)
    save_data(df, args.raw_output)
    
    # Prétraitement des données
    preprocess_data(args.raw_output, args.processed_output)

if __name__ == '__main__':
    main()
