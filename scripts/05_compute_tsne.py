from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
RANDOM_STATE = 42

train = pd.read_parquet(ARTIFACTS / "train.parquet")
with (ARTIFACTS / "features.json").open() as file:
    features = json.load(file)

X_train = train[features].to_numpy()
X_pca = PCA(n_components=30, random_state=RANDOM_STATE).fit_transform(X_train)
embedding = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
).fit_transform(X_pca)

np.save(ARTIFACTS / "tsne_embedding.npy", embedding)
print("Embedding t-SNE:", embedding.shape)
