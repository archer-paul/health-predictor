# Structure du projet HealthPredictor

```
HealthPredictor/
├── data/
│   ├── raw/                      # Données brutes
│   │   └── hospital_data.csv     # Données d'affluence hospitalière
│   └── processed/                # Données prétraitées 
├── models/                       # Dossier pour sauvegarder les modèles entraînés
├── notebooks/                    # Notebooks Jupyter pour l'exploration et la visualisation
│   ├── data_exploration.ipynb
│   ├── model_evaluation.ipynb
│   └── visualization.ipynb
├── src/                          # Code source
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py       # Génération de données d'enbtrainements et prétraitement de celles-civenv\Scripts\activate
│   │   └── data_loader.py        # Fonctions pour charger les données
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py     # Création de features pour les modèles
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima_model.py        # Modèle ARIMA
│   │   ├── lstm_model.py         # Modèle LSTM
│   │   └── train_model.py        # Script d'entraînement
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py          # Fonctions de visualisation
├── app/
│   ├── app.py                    # Application Flask pour le dashboard
│   ├── static/                   # Fichiers statiques (CSS, JS)
│   └── templates/                # Templates HTML
├── README.md                     # Documentation du projet
├── requirements.txt              # Dépendances Python
├── project_structure.py          # tructure du projet
└── setup.py                      # Script d'installation
```
