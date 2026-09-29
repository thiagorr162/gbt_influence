from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
TARGET = "refractive_index"
ID_COLUMN = "ID"
RANDOM_STATE = 42

train = pd.read_parquet(ARTIFACTS / "train.parquet")
test = pd.read_parquet(ARTIFACTS / "test.parquet")
with (ARTIFACTS / "features.json").open() as file:
    features = json.load(file)

X_train = train[features].to_numpy()
y_train = train[TARGET].to_numpy()
X_test = test[features].to_numpy()
y_test = test[TARGET].to_numpy()

model = LGBMRegressor(
    objective="regression",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)
model.fit(X_train, y_train)
joblib.dump(model, ARTIFACTS / "model.joblib")

prediction = model.predict(X_test)
metrics = {
    "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
    "mae": float(mean_absolute_error(y_test, prediction)),
}
with (ARTIFACTS / "metrics.json").open("w") as file:
    json.dump(metrics, file, indent=2)

predictions = pd.DataFrame({
    "test_position": np.arange(len(test)),
    ID_COLUMN: test[ID_COLUMN],
    "real": y_test,
    "prediction": prediction,
})
predictions.to_parquet(ARTIFACTS / "predictions.parquet", index=False)

print(f"RMSE: {metrics['rmse']:.5f}")
print(f"MAE:  {metrics['mae']:.5f}")
