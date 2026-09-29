from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
TARGET = "refractive_index"
ID_COLUMN = "ID"
RANDOM_STATE = 42

raw = pd.read_parquet(ROOT / "data" / "refractive_index.parquet")
if ID_COLUMN not in raw.columns:
    raw = raw.reset_index()

data = raw[raw[TARGET] <= 3].reset_index(drop=True)
features = [column for column in data.columns if column not in [ID_COLUMN, TARGET]]
X = data[features].to_numpy()
y = data[TARGET].to_numpy()

X_train, X_valid, y_train, y_valid, id_train, id_valid = train_test_split(
    X,
    y,
    data[ID_COLUMN].to_numpy(),
    test_size=0.2,
    random_state=RANDOM_STATE,
)

parameter_grid = {
    "n_estimators": [250, 400],
    "learning_rate": [0.03, 0.06],
    "num_leaves": [31, 63],
    "min_child_samples": [30, 70],
    "reg_lambda": [1.0],
}
combinations = list(ParameterGrid(parameter_grid))

results = []
best_parameters = None
best_rmse = np.inf
best_mae = np.inf
best_predictions = None

for number, parameters in enumerate(combinations, start=1):
    model = LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
        **parameters,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))
    mae = float(mean_absolute_error(y_valid, predictions))
    results.append({"combination": number, "rmse": rmse, "mae": mae, **parameters})
    print(f"Grid {number:02d}/{len(combinations)}: RMSE={rmse:.5f}, MAE={mae:.5f}, {parameters}", flush=True)
    if rmse < best_rmse:
        best_parameters = parameters.copy()
        best_rmse = rmse
        best_mae = mae
        best_predictions = predictions.copy()

results = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
results.to_csv(ARTIFACTS / "model_selection.csv", index=False)

model = LGBMRegressor(
    objective="regression",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
    **best_parameters,
)
model.fit(X, y)
joblib.dump(model, ARTIFACTS / "model.joblib")
data.to_parquet(ARTIFACTS / "training_data.parquet", index=False)
with (ARTIFACTS / "features.json").open("w") as file:
    json.dump(features, file, indent=2)

validation = pd.DataFrame({
    ID_COLUMN: id_valid,
    "real": y_valid,
    "prediction": best_predictions,
})
validation.to_parquet(ARTIFACTS / "validation_predictions.parquet", index=False)

metadata = {
    "target": TARGET,
    "id_column": ID_COLUMN,
    "random_state": RANDOM_STATE,
    "rows": len(data),
    "features": len(features),
    "grid_size": len(combinations),
    "validation_rmse": best_rmse,
    "validation_mae": best_mae,
    "parameters": best_parameters,
}
with (ARTIFACTS / "metadata.json").open("w") as file:
    json.dump(metadata, file, indent=2)


print(f"Melhores parâmetros: {best_parameters}", flush=True)
print(f"RMSE de validação: {best_rmse:.5f}", flush=True)
print(f"Modelo final retreinado em {len(data)} amostras", flush=True)
print(f"Artefatos salvos em: {ARTIFACTS}", flush=True)
