from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42

training = pd.read_parquet(ARTIFACTS / "training_data.parquet")
with (ARTIFACTS / "features.json").open() as file:
    features = json.load(file)

X = training[features].to_numpy()
X_pca = PCA(n_components=30, random_state=RANDOM_STATE).fit_transform(X)
embedding = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=RANDOM_STATE,
    n_jobs=-1,
).fit_transform(X_pca)
np.save(ARTIFACTS / "tsne_embedding.npy", embedding)
print(f"t-SNE salvo com shape {embedding.shape}", flush=True)
