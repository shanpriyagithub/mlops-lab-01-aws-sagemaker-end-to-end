# scripts/train_with_mlflow.py

import os
import json
import joblib
import pandas as pd
import xgboost as xgb


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")

def get_csv_path(train_dir: str) -> str:
    csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError(f"No CSV found in {train_dir}")
    return os.path.join(train_dir, csv_files[0])

def try_mlflow_setup(uri: str):
    if not uri:
        return None
    try:
        import mlflow
        import mlflow.xgboost
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("wine-quality-training")
        return mlflow
    except Exception as e:
        print(f"[WARN] MLflow setup failed, continuing without MLflow. Error: {e}")
        return None

csv_path = get_csv_path(TRAIN_DIR)
df = pd.read_csv(csv_path)

# target = last column (wine quality)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

mlflow = try_mlflow_setup(MLFLOW_URI)

if mlflow:
    try:
        with mlflow.start_run():
            mlflow.log_params(params)

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)

            mlflow.log_metric("rmse", float(rmse))
            mlflow.log_metric("r2", float(r2))

            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

            metrics_path = os.path.join(MODEL_DIR, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"rmse": float(rmse), "r2": float(r2)}, f)

            mlflow.log_artifact(metrics_path)
            mlflow.xgboost.log_model(model, artifact_path="model")

        print("[OK] Trained + logged to MLflow")
    except Exception as e:
        print(f"[WARN] MLflow logging failed mid-run, continuing. Error: {e}")

        # still save model so job succeeds
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
        print("[OK] Trained model saved (without MLflow)")
else:
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
    print("[OK] Trained model saved (MLflow disabled)")
