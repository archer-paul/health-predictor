#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Flask pour le dashboard de visualisation des tendances sanitaires
et des prédictions d'affluence hospitalière.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import pickle

from flask import Flask, render_template, request, jsonify, redirect, url_for
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ajout du répertoire parent au path pour importer les modules du projet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import des modules personnalisés
from src.models.arima_model import ArimaModel
from src.models.lstm_model import LSTMModel
from src.data.data_loader import load_data
from src.visualization.visualize import (
    plot_time_series, 
    plot_seasonal_patterns,
    plot_predictions_comparison, 
    create_interactive_dashboard
)

# Création de l'application Flask
app = Flask(__name__)

# Configuration
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
MODELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
TARGET_COL = 'patient_count'
DATE_COL = 'date'

# Chargement des données
def load_app_data():
    """
    Charge les données nécessaires pour l'application.
    
    Returns:
        tuple: (df_raw, df_processed) - DataFrames bruts et prétraités
    """
    raw_data_path = os.path.join(DATA_PATH, 'raw/hospital_data.csv')
    processed_data_path = os.path.join(DATA_PATH, 'processed/processed_hospital_data.csv')
    
    try:
        df_raw = pd.read_csv(raw_data_path)
        if DATE_COL in df_raw.columns:
            df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])
    except FileNotFoundError:
        df_raw = None
    
    try:
        df_processed = pd.read_csv(processed_data_path)
        if DATE_COL in df_processed.columns:
            df_processed[DATE_COL] = pd.to_datetime(df_processed[DATE_COL])
            
        # Extraction des composantes temporelles si elles n'existent pas déjà
        if 'month' not in df_processed.columns:
            df_processed['year'] = df_processed[DATE_COL].dt.year
            df_processed['month'] = df_processed[DATE_COL].dt.month
            df_processed['day_of_week'] = df_processed[DATE_COL].dt.dayofweek
            df_processed['quarter'] = df_processed[DATE_COL].dt.quarter
    except FileNotFoundError:
        df_processed = None
    
    return df_raw, df_processed

# Chargement des modèles entraînés
def load_models():
    """
    Charge les modèles ARIMA et LSTM entraînés.
    
    Returns:
        tuple: (arima_model, lstm_model) - Modèles chargés ou None si non disponibles
    """
    # Recherche du dernier modèle ARIMA
    arima_models = [f for f in os.listdir(MODELS_PATH) if f.startswith('arima_model_') and f.endswith('.pkl')]
    arima_model = None
    if arima_models:
        latest_arima = max(arima_models)
        try:
            with open(os.path.join(MODELS_PATH, latest_arima), 'rb') as f:
                arima_model = pickle.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle ARIMA: {e}")
    
    # Recherche du dernier modèle LSTM
    lstm_models = [d for d in os.listdir(MODELS_PATH) if d.startswith('lstm_model_') and os.path.isdir(os.path.join(MODELS_PATH, d))]
    lstm_model = None
    if lstm_models:
        latest_lstm = max(lstm_models)
        try:
            lstm_model = LSTMModel.load_model(os.path.join(MODELS_PATH, latest_lstm))
        except Exception as e:
            print(f"Erreur lors du chargement du modèle LSTM: {e}")
    
    return arima_model, lstm_model

# Route principale
@app.route('/')
def index():
    """
    Page d'accueil du dashboard.
    """
    return render_template('index.html')

# Route pour les données de tendances générales
@app.route('/api/trends')
def get_trends():
    """
    API pour obtenir les données de tendances générales.
    """
    _, df_processed = load_app_data()
    
    if df_processed is None:
        return jsonify({'error': 'Données non disponibles'})
    
    # Préparation des données pour le graphique d'évolution temporelle
    time_series = plot_time_series(df_processed, TARGET_COL, DATE_COL)
    time_series_json = json.dumps(time_series, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Préparation des patterns saisonniers
    monthly_fig, weekly_fig = plot_seasonal_patterns(df_processed, TARGET_COL, DATE_COL)
    monthly_json = json.dumps(monthly_fig, cls=plotly.utils.PlotlyJSONEncoder)
    weekly_json = json.dumps(weekly_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'time_series': time_series_json,
        'monthly_pattern': monthly_json,
        'weekly_pattern': weekly_json
    })

# Route pour le dashboard complet
@app.route('/api/dashboard')
def get_dashboard():
    """
    API pour obtenir le dashboard complet.
    """
    _, df_processed = load_app_data()
    
    if df_processed is None:
        return jsonify({'error': 'Données non disponibles'})
    
    # Création du dashboard interactif
    dashboard_fig = create_interactive_dashboard(df_processed, TARGET_COL)
    dashboard_json = json.dumps(dashboard_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({'dashboard': dashboard_json})

# Route pour les prédictions
@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    """
    Page de prédictions avec formulaire pour choisir la période.
    """
    if request.method == 'POST':
        # Récupération des paramètres du formulaire
        start_date = request.form.get('start_date')
        forecast_days = int(request.form.get('forecast_days', 30))
        
        return redirect(url_for('show_predictions', start_date=start_date, forecast_days=forecast_days))
    
    return render_template('predictions.html')

# Route pour afficher les prédictions
@app.route('/predictions/show')
def show_predictions():
    """
    Affiche les prédictions pour la période spécifiée.
    """
    start_date = request.args.get('start_date')
    forecast_days = int(request.args.get('forecast_days', 30))
    
    if not start_date:
        # Par défaut, utiliser la date actuelle
        start_date = datetime.now().strftime('%Y-%m-%d')
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = start_date + timedelta(days=forecast_days)
    
    # Chargement des modèles
    arima_model, lstm_model = load_models()
    
    if arima_model is None and lstm_model is None:
        return render_template('predictions.html', error="Aucun modèle disponible pour les prédictions")
    
    # Chargement des données
    _, df_processed = load_app_data()
    
    if df_processed is None:
        return render_template('predictions.html', error="Données non disponibles pour les prédictions")
    
    # Préparation des dates pour les prédictions
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Création du DataFrame pour les prédictions
    predictions_df = pd.DataFrame({
        'date': prediction_dates
    })
    
    # Génération des prédictions avec les modèles disponibles
    predictions = {}
    
    if arima_model is not None:
        try:
            # Prédictions avec ARIMA
            arima_preds = arima_model.predict(len(prediction_dates))
            predictions['ARIMA'] = arima_preds[:len(prediction_dates)]
        except Exception as e:
            print(f"Erreur lors des prédictions ARIMA: {e}")
    
    if lstm_model is not None:
        try:
            # Prédictions avec LSTM
            lstm_preds = lstm_model.predict(df_processed, len(prediction_dates))
            predictions['LSTM'] = lstm_preds[:len(prediction_dates)]
        except Exception as e:
            print(f"Erreur lors des prédictions LSTM: {e}")
    
    # Création du graphique de comparaison des prédictions
    if predictions:
        for model_name, preds in predictions.items():
            predictions_df[model_name] = preds
        
        fig = plot_predictions_comparison(predictions_df, DATE_COL, predictions.keys())
        predictions_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return render_template(
            'show_predictions.html',
            predictions=predictions_json,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            forecast_days=forecast_days
        )
    else:
        return render_template('predictions.html', error="Erreur lors de la génération des prédictions")

# Route pour obtenir les comparaisons des performances des modèles
@app.route('/api/model_performance')
def get_model_performance():
    """
    API pour obtenir les métriques de performance des modèles.
    """
    _, df_processed = load_app_data()
    
    if df_processed is None:
        return jsonify({'error': 'Données non disponibles'})
    
    # Chargement des métriques de performance
    try:
        performance_path = os.path.join(MODELS_PATH, 'model_performance.json')
        with open(performance_path, 'r') as f:
            performance_metrics = json.load(f)
        
        # Création du graphique de comparaison des performances
        models = list(performance_metrics.keys())
        metrics = ['MAE', 'RMSE', 'MAPE']
        
        fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metrics)
        
        for i, metric in enumerate(metrics):
            values = [performance_metrics[model].get(metric, 0) for model in models]
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=1, col=i+1
            )
        
        fig.update_layout(height=400, width=800, title_text="Comparaison des performances des modèles")
        
        return jsonify({
            'metrics': performance_metrics,
            'chart': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        })
    except Exception as e:
        return jsonify({'error': f'Erreur lors du chargement des métriques: {e}'})

# Route pour l'analyse des tendances
@app.route('/trends')
def trends():
    """
    Page d'analyse des tendances.
    """
    return render_template('trends.html')

# Route pour la page d'exploration des données
@app.route('/explore')
def explore():
    """
    Page d'exploration des données.
    """
    _, df_processed = load_app_data()
    
    if df_processed is None:
        return render_template('explore.html', error="Données non disponibles")
    
    # Statistiques descriptives
    stats = df_processed.describe().to_html(classes='table table-striped')
    
    # Dernières données
    recent_data = df_processed.tail(10).to_html(classes='table table-striped')
    
    return render_template('explore.html', stats=stats, recent_data=recent_data)

# Point d'entrée de l'application
if __name__ == '__main__':
    app.run(debug=True)
