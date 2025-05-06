#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module pour la visualisation des tendances sanitaires et des prédictions.
Crée des visualisations interactives pour identifier les patterns saisonniers et hebdomadaires.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

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

def plot_time_series(df, target_col, date_col='date', output_dir=None, filename="time_series.html"):
    """
    Crée une visualisation interactive de la série temporelle.
    
    Args:
        df: DataFrame pandas avec les données
        target_col: Nom de la colonne cible à visualiser
        date_col: Nom de la colonne contenant les dates
        output_dir: Répertoire où sauvegarder la visualisation
        filename: Nom du fichier de sortie
    """
    logger.info(f"Création de la visualisation de série temporelle pour {target_col}")
    
    # Création du graphique interactif
    fig = px.line(
        df, 
        x=date_col, 
        y=target_col,
        title=f"Évolution du {target_col} au fil du temps",
        labels={target_col: "Nombre de patients", date_col: "Date"},
        template="plotly_white"
    )
    
    # Ajout de la moyenne mobile sur 7 jours
    if 'rolling_mean_7d' in df.columns:
        fig.add_scatter(
            x=df[date_col], 
            y=df['rolling_mean_7d'],
            mode='lines',
            name='Moyenne mobile (7 jours)',
            line=dict(color='red', width=2)
        )
    
    # Mise en forme
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Nombre de patients",
        legend_title="Métrique",
        hovermode="x unified"
    )
    
    # Sauvegarde de la visualisation
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, filename))
        logger.info(f"Visualisation sauvegardée dans {os.path.join(output_dir, filename)}")
    
    return fig

def plot_seasonal_patterns(df, target_col, date_col='date', output_dir=None):
    """
    Visualise les patterns saisonniers dans les données.
    
    Args:
        df: DataFrame pandas avec les données
        target_col: Nom de la colonne cible à visualiser
        date_col: Nom de la colonne contenant les dates
        output_dir: Répertoire où sauvegarder les visualisations
    """
    logger.info("Création des visualisations de patterns saisonniers")
    
    # Extraction des composantes temporelles si elles n'existent pas déjà
    if 'month' not in df.columns:
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
    
    # Création du répertoire de sortie si nécessaire
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pattern mensuel
    monthly_avg = df.groupby('month')[target_col].mean().reset_index()
    monthly_fig = px.bar(
        monthly_avg,
        x='month',
        y=target_col,
        title=f"Pattern mensuel de {target_col}",
        labels={target_col: "Moyenne de patients", 'month': "Mois"},
        template="plotly_white"
    )
    monthly_fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(1, 13))))
    
    if output_dir:
        monthly_fig.write_html(os.path.join(output_dir, "monthly_pattern.html"))
    
    # 2. Pattern hebdomadaire
    weekly_avg = df.groupby('day_of_week')[target_col].mean().reset_index()
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    weekly_avg['day_name'] = weekly_avg['day_of_week'].apply(lambda x: days[x])
    
    weekly_fig = px.bar(
        weekly_avg,
        x='day_name',
        y=target_col,
        title=f"Pattern hebdomadaire de {target_col}",
        labels={target_col: "Moyenne de patients", 'day_name': "Jour"},
        template="plotly_white",
        category_orders={"day_name": days}
    )
    
    if output_dir:
        weekly_fig.write_html(os.path.join(output_dir, "weekly_pattern.html"))
    
    # 3. Heatmap combiné (jour de semaine x mois)
    if len(df) > 100:  # Suffisamment de données pour cette analyse
        pivot_data = df.pivot_table(
            index='day_of_week',
            columns='month',
            values=target_col,
            aggfunc='mean'
        )
        
        heatmap_fig = px.imshow(
            pivot_data,
            labels=dict(x="Mois", y="Jour de la semaine", color="Moyenne de patients"),
            x=[str(i) for i in range(1, 13)],
            y=days,
            title=f"Heatmap des patterns de {target_col} (Mois x Jour de semaine)",
            color_continuous_scale="YlOrRd"
        )
        
        if output_dir:
            heatmap_fig.write_html(os.path.join(output_dir, "heatmap_pattern.html"))
    
    logger.info("Visualisations des patterns saisonniers créées avec succès")
    
    return monthly_fig, weekly_fig

def plot_predictions_comparison(y_true, y_pred_arima, y_pred_lstm, dates=None, output_dir=None):
    """
    Compare les prédictions des modèles ARIMA et LSTM avec les valeurs réelles.
    
    Args:
        y_true: Valeurs réelles
        y_pred_arima: Prédictions du modèle ARIMA
        y_pred_lstm: Prédictions du modèle LSTM
        dates: Dates correspondantes (optionnel)
        output_dir: Répertoire où sauvegarder la visualisation
    """
    logger.info("Création de la visualisation comparative des prédictions")
    
    # Préparation des données pour le tracé
    if dates is None:
        x_values = np.arange(len(y_true))
    else:
        x_values = dates
    
    # Création du graphique comparatif
    fig = go.Figure()
    
    # Ajout des valeurs réelles
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_true,
        mode='lines+markers',
        name='Valeurs réelles',
        line=dict(color='black', width=2)
    ))
    
    # Ajout des prédictions ARIMA
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_pred_arima,
        mode='lines',
        name='Prédictions ARIMA',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Ajout des prédictions LSTM
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_pred_lstm,
        mode='lines',
        name='Prédictions LSTM',
        line=dict(color='red', width=2, dash='dot')
    ))
    
    # Mise en forme
    fig.update_layout(
        title="Comparaison des prédictions ARIMA et LSTM",
        xaxis_title="Date" if dates is not None else "Période",
        yaxis_title="Nombre de patients",
        legend_title="Modèle",
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Sauvegarde de la visualisation
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, "predictions_comparison.html"))
        logger.info(f"Visualisation comparative sauvegardée dans {os.path.join(output_dir, 'predictions_comparison.html')}")
    
    return fig

def plot_feature_importance(feature_names, importances, output_dir=None):
    """
    Visualise l'importance des caractéristiques pour les modèles.
    
    Args:
        feature_names: Noms des caractéristiques
        importances: Scores d'importance des caractéristiques
        output_dir: Répertoire où sauvegarder la visualisation
    """
    logger.info("Création de la visualisation d'importance des caractéristiques")
    
    # Création du DataFrame pour le tracé
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Création du graphique d'importance
    fig = px.bar(
        imp_df.head(15),  # Top 15 caractéristiques
        x='Importance',
        y='Feature',
        orientation='h',
        title="Importance des caractéristiques",
        labels={'Importance': "Score d'importance", 'Feature': "Caractéristique"},
        template="plotly_white"
    )
    
    # Mise en forme
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        hovermode="y unified"
    )
    
    # Sauvegarde de la visualisation
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, "feature_importance.html"))
        logger.info(f"Visualisation d'importance sauvegardée dans {os.path.join(output_dir, 'feature_importance.html')}")
    
    return fig

def create_interactive_dashboard(df, target_col, predictions_df=None, output_dir=None):
    """
    Crée un dashboard interactif combinant plusieurs visualisations.
    
    Args:
        df: DataFrame pandas avec les données
        target_col: Nom de la colonne cible
        predictions_df: DataFrame avec les prédictions (optionnel)
        output_dir: Répertoire où sauvegarder le dashboard
    """
    logger.info("Création du dashboard interactif")
    
    # Création de la figure principale avec sous-graphiques
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Évolution temporelle",
            "Pattern hebdomadaire",
            "Pattern mensuel",
            "Prédictions (si disponibles)"
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Évolution temporelle
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df[target_col],
            mode='lines',
            name=target_col
        ),
        row=1, col=1
    )
    
    # Ajout de la moyenne mobile sur 7 jours
    if 'rolling_mean_7d' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['rolling_mean_7d'],
                mode='lines',
                name='Moyenne mobile (7j)',
                line=dict(color='red')
            ),
            row=1, col=1
        )
    
    # 2. Pattern hebdomadaire
    weekly_avg = df.groupby('day_of_week')[target_col].mean().reset_index()
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    weekly_avg['day_name'] = weekly_avg['day_of_week'].apply(lambda x: days[x])
    
    fig.add_trace(
        go.Bar(
            x=weekly_avg['day_name'],
            y=weekly_avg[target_col],
            name="Moyenne hebdomadaire",
            marker_color='lightskyblue'
        ),
        row=1, col=2
    )
    
    # 3. Pattern mensuel
    monthly_avg = df.groupby('month')[target_col].mean().reset_index()
    
    fig.add_trace(
        go.Bar(
            x=monthly_avg['month'],
            y=monthly_avg[target_col],
            name="Moyenne mensuelle",
            marker_color='lightgreen'
        ),
        row=2, col=1
    )
    
    # 4. Prédictions (si disponibles)
    if predictions_df is not None and len(predictions_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=predictions_df['date'],
                y=predictions_df['actual'],
                mode='lines+markers',
                name='Valeurs réelles',
                line=dict(color='black')
            ),
            row=2, col=2
        )
        
        if 'arima_pred' in predictions_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions_df['date'],
                    y=predictions_df['arima_pred'],
                    mode='lines',
                    name='ARIMA',
                    line=dict(color='blue', dash='dash')
                ),
                row=2, col=2
            )
        
        if 'lstm_pred' in predictions_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions_df['date'],
                    y=predictions_df['lstm_pred'],
                    mode='lines',
                    name='LSTM',
                    line=dict(color='red', dash='dot')
                ),
                row=2, col=2
            )
    
    # Mise en forme globale
    fig.update_layout(
        title_text="Dashboard de tendances sanitaires",
        height=800,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Mise à jour des axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Nombre de patients", row=1, col=1)
    
    fig.update_xaxes(title_text="Jour de la semaine", row=1, col=2)
    fig.update_yaxes(title_text="Moyenne de patients", row=1, col=2)
    
    fig.update_xaxes(title_text="Mois", row=2, col=1)
    fig.update_yaxes(title_text="Moyenne de patients", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Nombre de patients", row=2, col=2)
    
    # Sauvegarde du dashboard
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(os.path.join(output_dir, "dashboard.html"))
        logger.info(f"Dashboard sauvegardé dans {os.path.join(output_dir, 'dashboard.html')}")
    
    return fig

def main(data_filepath='../../data/processed/processed_hospital_data.csv',
         target_col='patient_count',
         date_col='date',
         output_dir='../../reports/figures'):
    """
    Point d'entrée principal pour la création des visualisations.
    
    Args:
        data_filepath: Chemin vers le fichier de données
        target_col: Nom de la colonne cible à visualiser
        date_col: Nom de la colonne contenant les dates
        output_dir: Répertoire où sauvegarder les visualisations
    """
    # Chargement des données
    df = load_data(data_filepath)
    
    # Création des visualisations de base
    plot_time_series(df, target_col, date_col, output_dir)
    plot_seasonal_patterns(df, target_col, date_col, output_dir)
    
    # Création du dashboard interactif
    create_interactive_dashboard(df, target_col, None, output_dir)
    
    logger.info("Création des visualisations terminée avec succès")

if __name__ == '__main__':
    main()
