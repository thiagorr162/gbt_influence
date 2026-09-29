from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from tree_influence.explainers import BoostIn

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
TARGET = "refractive_index"
ID_COLUMN = "ID"
RANDOM_STATE = 42
POINTS_PER_QUANTILE = 50

train = pd.read_parquet(ARTIFACTS / "train.parquet")
test = pd.read_parquet(ARTIFACTS / "test.parquet")
model = joblib.load(ARTIFACTS / "model.joblib")
with (ARTIFACTS / "features.json").open() as file:
    features = json.load(file)

selection = pd.DataFrame({
    "test_position": np.arange(len(test)),
    ID_COLUMN: test[ID_COLUMN],
    TARGET: test[TARGET],
})
selection["ri_quantile"] = pd.qcut(selection[TARGET], q=10, labels=False)
selected = (
    selection.groupby("ri_quantile", group_keys=False)
    .sample(n=POINTS_PER_QUANTILE, random_state=RANDOM_STATE)
    .sort_values(["ri_quantile", TARGET])
    .reset_index(drop=True)
)
selected["influence_column"] = np.arange(len(selected))
selected.to_parquet(ARTIFACTS / "selected_targets.parquet", index=False)

summary = selection.groupby("ri_quantile")[TARGET].agg(
    total_points="count",
    min_ri="min",
    max_ri="max",
)
summary["selected_points"] = selected.groupby("ri_quantile").size()
summary.to_csv(ARTIFACTS / "ri_quantiles.csv")

X_train = train[features].to_numpy()
y_train = train[TARGET].to_numpy()
positions = selected["test_position"].to_numpy()
X_target = test.iloc[positions][features].to_numpy()
y_target = test.iloc[positions][TARGET].to_numpy()

boostin = BoostIn()
boostin.fit(model, X_train, y_train)
influence = boostin.get_local_influence(X_target, y_target, verbose=0)
np.save(ARTIFACTS / "boostin_influence.npy", influence)

print(summary)
print("Matriz de influência:", influence.shape)
