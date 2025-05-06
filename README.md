# HealthPredictor
This README is also available in [French 🇫🇷](README.fr.md)

## Medical Trend Prediction with Machine Learning

HealthPredictor is a predictive analytics tool designed to anticipate crowd surges in hospital facilities. The project uses advanced time series models (ARIMA, LSTM) to analyze historical hospital attendance data and forecast future trends.

## Features

- **Predictive analysis**: Uses ARIMA and LSTM algorithms to forecast hospital attendance trends
- **Pattern detection**: Automatically identifies seasonal and weekly trends
- **Interactive dashboard**: Visualizes historical data and predictions
- **Preventive alerts**: Generates alerts for anticipated crowd peaks

## Installation

```bash
# Clone the repository
git clone https://github.com/archer-paul/HealthPredictor.git
cd HealthPredictor

# Create a virtual environment
python -m venv env
source env/bin/activate  # For Linux/Mac
# or
env\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt
```

## Project structure
- `data/` : Raw and preprocessed data
- `notebooks/` : Jupyter notebooks for data exploration and model evaluation
- `src/` : Source code
  - `data/` : Scripts for data loading and preprocessing
  - `features/` : Feature extraction and preparation for models
  - `models/` : ARIMA and LSTM model implementations
  - `visualization/` : Tools for data and prediction visualization
  - `app/` : Interactive dashboard (Flask)

## Usage
### Data preprocessing
```bash
python src/data/make_dataset.py
```
### Model training
```bash
python src/models/train_model.py --model arima  # For the ARIMA model
python src/models/train_model.py --model lstm   # For the LSTM model
```
### Launching the dashboard
```bash
python app/app.py
```

Navigate to `http://localhost:5000` to access the dashboard.

## Results overview
The model effectively identifies seasonal and weekly patterns in hospital attendance data:

- **Seasonal trends**: Peaks during flu season (December–February)

- **Weekly trends**: Higher attendance on Mondays and Fridays
- **Special events**: Detection of peaks related to specific events (heatwaves, epidemics)

## Technologies used
- Python 3.8+
- Pandas, NumPy, Scikit-learn
- Statsmodels (ARIMA)
- TensorFlow/Keras (LSTM)
- Plotly and Matplotlib (visualization)
- Flask (web dashboard)

## License
MIT

## Contact
For any questions or suggestions, feel free to contact me: [paul.erwan.archer@gmail.com](mailto:paul.erwan.archer@gmail.com)