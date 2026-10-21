# EstateIQ — House Price Intelligence

An interactive machine learning application that predicts house prices using property features. Built with Python, Scikit-learn, and Streamlit as part of **Prodigy InfoTech Machine Learning Internship — Task 1**.

## Overview

EstateIQ compares machine learning regression models and provides an interactive dashboard for exploring housing data, reviewing model performance, and estimating house prices.

## Features

- House price prediction using property features
- Comparison of Linear Regression and Random Forest
- Interactive Streamlit dashboard
- Model performance metrics and visualizations
- Dataset exploration and prediction insights
- Saved trained model for predictions

## Machine Learning Models

- Linear Regression
- Random Forest Regressor

The models use three features: living area (`GrLivArea`), number of bedrooms (`BedroomAbvGr`), and number of full bathrooms (`FullBath`).

## Model Results

Evaluation on the held-out test set:

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | $35,788.06 | $33,264.92 |
| RMSE | $52,975.72 | $49,076.82 |
| R² Score | 0.6341 | 0.6860 |

Random Forest achieved the better R² score in this experiment and was selected as the best-performing model.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly and Matplotlib
- Joblib

## Project Structure

```text
House-Price-Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── train.csv
├── models/
│   └── house_price_model.joblib
└── outputs/
    ├── metrics.json
    ├── test_predictions.csv
    └── actual_vs_predicted.html
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/20pwele5276-jpg/PRODIGY_ML_01_House-Price-Prediction.git
   cd PRODIGY_ML_01_House-Price-Prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

If the trained model is unavailable, train it first:

```bash
python train_model.py
```

## Limitations

This project is for educational and demonstration purposes. The prediction uses only three property features and does not account for location, property condition, neighborhood, or other factors that influence real-world prices. Predictions should not be treated as formal property valuations.

## Internship

**Organization:** Prodigy InfoTech  
**Task:** Machine Learning Task 1 — House Price Prediction

## Author

**Wajiha Gul**  
[GitHub Profile](https://github.com/20pwele5276-jpg)
