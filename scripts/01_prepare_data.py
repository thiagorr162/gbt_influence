from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET = "refractive_index"
ID_COLUMN = "ID"

raw = pd.read_parquet(ROOT / "data" / "refractive_index.parquet")
if ID_COLUMN not in raw.columns:
    raw = raw.reset_index()

data = raw[raw[TARGET] <= 3].reset_index(drop=True)
features = [column for column in data.columns if column not in [ID_COLUMN, TARGET]]

train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_parquet(ARTIFACTS / "train.parquet", index=False)
test.to_parquet(ARTIFACTS / "test.parquet", index=False)
with (ARTIFACTS / "features.json").open("w") as file:
    json.dump(features, file, indent=2)

print("Total após RI <= 3:", len(data))
print("Treino:", len(train))
print("Teste:", len(test))
print("Features:", len(features))
