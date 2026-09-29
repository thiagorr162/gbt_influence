from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
TARGET = "refractive_index"

train = pd.read_parquet(ARTIFACTS / "train.parquet")
test = pd.read_parquet(ARTIFACTS / "test.parquet")
predictions = pd.read_parquet(ARTIFACTS / "predictions.parquet")
selected = pd.read_parquet(ARTIFACTS / "selected_targets.parquet")
quantile_summary = pd.read_csv(ARTIFACTS / "ri_quantiles.csv", index_col=0)
influence = np.load(ARTIFACTS / "boostin_influence.npy", mmap_mode="r")
embedding = np.load(ARTIFACTS / "tsne_embedding.npy")

selected_quantiles = selected["ri_quantile"].to_numpy()
quantile_values = np.sort(selected["ri_quantile"].unique())
combined_signed = np.asarray(influence.mean(axis=1))
combined_absolute = np.asarray(np.abs(influence).mean(axis=1))
signed_by_quantile = np.column_stack([
    np.asarray(influence[:, selected_quantiles == quantile].mean(axis=1))
    for quantile in quantile_values
])
absolute_by_quantile = np.column_stack([
    np.asarray(np.abs(influence[:, selected_quantiles == quantile]).mean(axis=1))
    for quantile in quantile_values
])


def symmetric_log_norm(values):
    values = np.asarray(values)
    absolute = np.abs(values)
    nonzero = absolute[absolute > 0]
    limit = float(absolute.max())
    if limit == 0:
        limit = 1.0
    linear_threshold = float(np.quantile(nonzero, 0.10)) if len(nonzero) else 1.0
    return SymLogNorm(
        linthresh=linear_threshold,
        vmin=-limit,
        vmax=limit,
        base=10,
    )


def linear_norm(values):
    limit = float(np.quantile(np.abs(values), 0.95))
    if limit == 0:
        limit = 1.0
    return Normalize(vmin=-limit, vmax=limit, clip=True)


def gray_background(axes):
    for ax in np.asarray(axes).flat:
        ax.set_facecolor("#eeeeee")


positions = selected["test_position"].to_numpy()
test_signed = np.asarray(influence.sum(axis=0))
test_absolute = np.asarray(np.abs(influence).sum(axis=0))
test_sizes = 25 + 100 * np.sqrt(test_absolute / test_absolute.max())
plot_limits = [
    min(predictions["real"].min(), predictions["prediction"].min()),
    max(predictions["real"].max(), predictions["prediction"].max()),
]

for scale, norm in [
    ("log", symmetric_log_norm(test_signed)),
    ("linear", linear_norm(test_signed)),
]:
    fig, ax = plt.subplots(figsize=(8, 7))
    if scale == "linear":
        gray_background([ax])
    ax.scatter(
        predictions["real"], predictions["prediction"],
        color="gray", s=12, alpha=0.25,
    )
    points = ax.scatter(
        test.iloc[positions][TARGET],
        predictions.iloc[positions]["prediction"],
        c=test_signed,
        cmap="RdBu",
        norm=norm,
        s=test_sizes,
        alpha=0.85,
    )
    ax.plot(plot_limits, plot_limits, "--", color="black")
    suffix = "" if scale == "log" else " — cor linear robusta (P95)"
    ax.set(
        xlabel="RI real",
        ylabel="RI predito",
        title="Predição versus valor real" + suffix,
    )
    fig.colorbar(
        points, ax=ax, label="Influência líquida recebida",
        extend="both" if scale == "linear" else "neither",
    )
    fig.tight_layout()
    filename = "prediction_vs_real_influence.png" if scale == "log" else "prediction_vs_real_influence_linear.png"
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)

combined_log_norm = symmetric_log_norm(combined_signed)
combined_linear_norm = linear_norm(combined_signed)
for scale, norm in [("log", combined_log_norm), ("linear", combined_linear_norm)]:
    fig, ax = plt.subplots(figsize=(10, 7))
    if scale == "linear":
        gray_background([ax])
    points = ax.scatter(
        train[TARGET], combined_signed,
        c=combined_signed, cmap="RdBu", norm=norm,
        s=8, alpha=0.55,
    )
    ax.axhline(0, color="black", linewidth=1)
    if scale == "log":
        ax.set_yscale("symlog", linthresh=combined_log_norm.linthresh)
    else:
        top_indices = np.argsort(combined_absolute)[-20:]
        ax.scatter(
            train.iloc[top_indices][TARGET], combined_signed[top_indices],
            facecolors="none", edgecolors="black", s=70, linewidths=1,
        )
    suffix = "symlog" if scale == "log" else "linear robusta (P95 da cor)"
    ax.set(
        xlabel="RI do ponto de treino",
        ylabel="Influência média",
        title=f"RI versus influência conjunta — escala {suffix}",
    )
    fig.colorbar(
        points, ax=ax, label="Influência média",
        extend="both" if scale == "linear" else "neither",
    )
    fig.tight_layout()
    filename = "ri_vs_influence.png" if scale == "log" else "ri_vs_influence_linear.png"
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)

all_quantile_log_norm = symmetric_log_norm(signed_by_quantile)
fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey=True)
for column, (ax, quantile) in enumerate(zip(axes.flat, quantile_values)):
    signed = signed_by_quantile[:, column]
    limits = quantile_summary.loc[quantile]
    ax.scatter(
        train[TARGET], signed,
        c=signed, cmap="RdBu", norm=all_quantile_log_norm,
        s=4, alpha=0.45,
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_yscale("symlog", linthresh=all_quantile_log_norm.linthresh)
    ax.set_title(f"Decil {quantile + 1}\nRI {limits.min_ri:.3f}–{limits.max_ri:.3f}")
for ax in axes[-1]:
    ax.set_xlabel("RI do treino")
for ax in axes[:, 0]:
    ax.set_ylabel("Influência média")
colorbar_axis = fig.add_axes([0.91, 0.15, 0.015, 0.68])
fig.colorbar(
    plt.cm.ScalarMappable(norm=all_quantile_log_norm, cmap="RdBu"),
    cax=colorbar_axis,
    label="Influência média",
)
fig.suptitle("RI versus influência por decil — escala symlog global", fontsize=16)
fig.subplots_adjust(top=0.88, right=0.89, wspace=0.15, hspace=0.28)
fig.savefig(FIGURES / "ri_vs_influence_by_decile.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(27, 10), sharex=True, constrained_layout=True)
gray_background(axes)
for column, (ax, quantile) in enumerate(zip(axes.flat, quantile_values)):
    signed = signed_by_quantile[:, column]
    limits = quantile_summary.loc[quantile]
    norm = linear_norm(signed)
    points = ax.scatter(
        train[TARGET], signed,
        c=signed, cmap="RdBu", norm=norm,
        s=4, alpha=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.7)
    top_indices = np.argsort(np.abs(signed))[-10:]
    ax.scatter(
        train.iloc[top_indices][TARGET], signed[top_indices],
        facecolors="none", edgecolors="black", s=45, linewidths=0.8,
    )
    ax.set_title(f"Decil {quantile + 1}\nRI {limits.min_ri:.3f}–{limits.max_ri:.3f}")
    fig.colorbar(
        points, ax=ax, fraction=0.046, pad=0.02,
        format="%.1e", extend="both",
    )
for ax in axes[-1]:
    ax.set_xlabel("RI do treino")
for ax in axes[:, 0]:
    ax.set_ylabel("Influência média")
fig.suptitle("RI versus influência por decil — cor linear robusta (P95 por painel)", fontsize=16)
fig.savefig(FIGURES / "ri_vs_influence_by_decile_linear.png", dpi=180)
plt.close(fig)

for scale, norm in [("log", combined_log_norm), ("linear", combined_linear_norm)]:
    fig, ax = plt.subplots(figsize=(11, 8))
    if scale == "linear":
        gray_background([ax])
    points = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=combined_signed, cmap="RdBu", norm=norm,
        s=8, alpha=0.65,
    )
    top_indices = np.argsort(combined_absolute)[-20:]
    ax.scatter(
        embedding[top_indices, 0], embedding[top_indices, 1],
        facecolors="none", edgecolors="black", s=90, linewidths=1.5,
    )
    suffix = "log simétrica" if scale == "log" else "linear robusta (P95)"
    ax.set(
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        title=f"t-SNE da influência conjunta — cor em escala {suffix}",
    )
    fig.colorbar(
        points, ax=ax, label="Influência média",
        extend="both" if scale == "linear" else "neither",
    )
    fig.tight_layout()
    filename = "tsne_influence.png" if scale == "log" else "tsne_influence_linear.png"
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(22, 10), sharex=True, sharey=True)
for column, (ax, quantile) in enumerate(zip(axes.flat, quantile_values)):
    signed = signed_by_quantile[:, column]
    absolute = absolute_by_quantile[:, column]
    top_indices = np.argsort(absolute)[-10:]
    limits = quantile_summary.loc[quantile]
    ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=signed, cmap="RdBu", norm=all_quantile_log_norm,
        s=4, alpha=0.6,
    )
    ax.scatter(
        embedding[top_indices, 0], embedding[top_indices, 1],
        facecolors="none", edgecolors="black", s=50, linewidths=1,
    )
    ax.set_title(f"Decil {quantile + 1}\nRI {limits.min_ri:.3f}–{limits.max_ri:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
colorbar_axis = fig.add_axes([0.91, 0.15, 0.015, 0.68])
fig.colorbar(
    plt.cm.ScalarMappable(norm=all_quantile_log_norm, cmap="RdBu"),
    cax=colorbar_axis,
    label="Influência média",
)
fig.suptitle("t-SNE da influência por decil — escala log simétrica global", fontsize=16)
fig.subplots_adjust(top=0.88, right=0.89, wspace=0.05, hspace=0.12)
fig.savefig(FIGURES / "tsne_influence_by_decile.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(22, 10), sharex=True, sharey=True, constrained_layout=True)
gray_background(axes)
for column, (ax, quantile) in enumerate(zip(axes.flat, quantile_values)):
    signed = signed_by_quantile[:, column]
    absolute = absolute_by_quantile[:, column]
    top_indices = np.argsort(absolute)[-10:]
    limits = quantile_summary.loc[quantile]
    norm = linear_norm(signed)
    points = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=signed, cmap="RdBu", norm=norm,
        s=4, alpha=0.65,
    )
    ax.scatter(
        embedding[top_indices, 0], embedding[top_indices, 1],
        facecolors="none", edgecolors="black", s=50, linewidths=1,
    )
    ax.set_title(f"Decil {quantile + 1}\nRI {limits.min_ri:.3f}–{limits.max_ri:.3f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(
        points, ax=ax, fraction=0.046, pad=0.02,
        format="%.1e", extend="both",
    )
fig.suptitle("t-SNE da influência por decil — cor linear robusta (P95 por painel)", fontsize=16)
fig.savefig(FIGURES / "tsne_influence_by_decile_linear.png", dpi=180)
plt.close(fig)

print("Figuras salvas em:", FIGURES)
for path in sorted(FIGURES.glob("*.png")):
    print("-", path.name)
