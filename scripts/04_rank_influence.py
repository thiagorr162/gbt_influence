from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
TARGET = "refractive_index"
ID_COLUMN = "ID"

train = pd.read_parquet(ARTIFACTS / "train.parquet")
selected = pd.read_parquet(ARTIFACTS / "selected_targets.parquet")
influence = np.load(ARTIFACTS / "boostin_influence.npy", mmap_mode="r")

mean_influence = np.asarray(influence.mean(axis=1))
mean_absolute = np.asarray(np.abs(influence).mean(axis=1))
ranking = pd.DataFrame({
    ID_COLUMN: train[ID_COLUMN],
    TARGET: train[TARGET],
    "mean_influence": mean_influence,
    "mean_absolute_influence": mean_absolute,
    "type": np.where(mean_influence >= 0, "proponent", "opponent"),
})
ranking = ranking.sort_values("mean_absolute_influence", ascending=False).reset_index(drop=True)
ranking["rank"] = np.arange(1, len(ranking) + 1)
ranking.to_csv(ARTIFACTS / "boostin_ranking.csv", index=False)
ranking.to_parquet(ARTIFACTS / "boostin_ranking.parquet", index=False)

rankings = []
for quantile in sorted(selected["ri_quantile"].unique()):
    columns = selected.index[selected["ri_quantile"] == quantile].to_numpy()
    signed = np.asarray(influence[:, columns].mean(axis=1))
    absolute = np.asarray(np.abs(influence[:, columns]).mean(axis=1))
    quantile_ranking = pd.DataFrame({
        "ri_quantile": int(quantile) + 1,
        ID_COLUMN: train[ID_COLUMN],
        TARGET: train[TARGET],
        "mean_influence": signed,
        "mean_absolute_influence": absolute,
        "type": np.where(signed >= 0, "proponent", "opponent"),
    })
    quantile_ranking = quantile_ranking.sort_values(
        "mean_absolute_influence", ascending=False,
    ).reset_index(drop=True)
    quantile_ranking["rank"] = np.arange(1, len(quantile_ranking) + 1)
    rankings.append(quantile_ranking)

ranking_by_quantile = pd.concat(rankings, ignore_index=True)
ranking_by_quantile.to_parquet(
    ARTIFACTS / "boostin_ranking_by_ri_decile.parquet", index=False,
)
ranking_by_quantile.groupby("ri_quantile", sort=True).head(20).to_csv(
    ARTIFACTS / "boostin_top20_by_ri_decile.csv", index=False,
)

print("Ranking conjunto:", ranking.shape)
print("Ranking por decil:", ranking_by_quantile.shape)
