
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURES = ["GrLivArea", "BedroomAbvGr", "FullBath"]
TARGET = "SalePrice"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    data = data[FEATURES + [TARGET]].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET] > 0]

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    candidates = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    results = []
    best_model = None
    best_name = None
    best_r2 = float("-inf")
    best_predictions = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4),
        })

        print(f"\n{name}")
        print("-" * 30)
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:    {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
            best_predictions = predictions

    joblib.dump(best_model, MODEL_DIR / "house_price_model.joblib")

    metrics = {
        "model": best_name,
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "mae": round(float(mean_absolute_error(y_test, best_predictions)), 2),
        "rmse": round(float(mean_squared_error(y_test, best_predictions) ** 0.5), 2),
        "r2": round(float(best_r2), 4),
        "model_comparison": results,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    predictions_df = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": best_predictions.round(2),
    })
    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False
    )

    chart = px.scatter(
        predictions_df,
        x="Actual Price",
        y="Predicted Price",
        title=f"Actual vs. Predicted House Prices — {best_name}",
        opacity=0.75,
        template="plotly_white",
    )

    lower = float(min(predictions_df.min()))
    upper = float(max(predictions_df.max()))
    chart.add_shape(
        type="line",
        x0=lower, y0=lower,
        x1=upper, y1=upper,
        line={"dash": "dash", "color": "red"},
    )
    chart.write_html(
        OUTPUT_DIR / "actual_vs_predicted.html",
        include_plotlyjs=True,
    )

    print("\n" + "=" * 42)
    print(f"BEST MODEL: {best_name}")
    print(f"Best test R²: {best_r2:.4f}")
    print(f"Saved model: {MODEL_DIR / 'house_price_model.joblib'}")
    print("Training and comparison completed successfully.")


if __name__ == "__main__":
    main()
