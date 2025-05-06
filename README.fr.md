# HealthPredictor
Ce README est aussi disponible en [anglais 🇬🇧](README.md)

## Prédiction de tendances médicales avec Machine Learning

HealthPredictor est un outil d'analyse prédictive conçu pour anticiper les pics d'affluence dans les établissements hospitaliers. Le projet utilise des modèles de séries temporelles avancés (ARIMA, LSTM) pour analyser des données historiques de fréquentation hospitalière et prévoir les tendances futures.

## Caractéristiques

- **Analyse prédictive** : Utilisation d'algorithmes ARIMA et LSTM pour prédire les tendances d'affluence hospitalière
- **Identification de patterns** : Détection automatique des tendances saisonnières et hebdomadaires
- **Dashboard interactif** : Visualisation des données historiques et des prévisions
- **Alertes préventives** : Génération d'alertes pour les pics d'affluence prévus

## Installation

```bash
# Clonage du dépôt
git clone https://github.com/archer-paul/HealthPredictor.git
cd HealthPredictor

# Création d'un environnement virtuel
python -m venv env
source env/bin/activate  # Pour Linux/Mac
# ou
env\Scripts\activate     # Pour Windows

# Installation des dépendances
pip install -r requirements.txt
```

## Structure du projet

- `data/` : Données brutes et prétraitées
- `notebooks/` : Jupyter notebooks pour l'exploration de données et l'évaluation des modèles
- `src/` : Code source
  - `data/` : Scripts pour le chargement et le prétraitement des données
  - `features/` : Extraction et préparation de features pour les modèles
  - `models/` : Implémentation des modèles ARIMA et LSTM
  - `visualization/` : Outils de visualisation des données et des prédictions
- `app/` : Dashboard interactif (Flask)

## Utilisation

### Prétraitement des données

```bash
python src/data/make_dataset.py
```

### Entraînement des modèles

```bash
python src/models/train_model.py --model arima  # Pour le modèle ARIMA
python src/models/train_model.py --model lstm   # Pour le modèle LSTM
```

### Lancement du dashboard

```bash
python app/app.py
```

Naviguez vers `http://localhost:5000` pour accéder au dashboard.

## Aperçu des résultats

Le modèle identifie efficacement les patterns saisonniers et hebdomadaires dans les données d'affluence hospitalière:

- **Tendances saisonnières** : Pics durant la saison grippale (décembre-février)
- **Tendances hebdomadaires** : Fréquentation plus élevée les lundis et vendredis
- **Événements spéciaux** : Détection de pics liés à des événements spécifiques (canicules, épidémies)

## Technologies utilisées

- Python 3.8+
- Pandas, NumPy, Scikit-learn
- Statsmodels (ARIMA)
- TensorFlow/Keras (LSTM)
- Plotly et Matplotlib (visualisation)
- Flask (dashboard web)

## Licence

MIT

## Contact

Pour toute question ou suggestion, n'hésitez pas à me contacter : [paul.erwan.archer@gmail.com](mailto:paul.erwan.archer@gmail.com)
